# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Actual native shared-memory setup under mocked CUDA capacities/contexts.

The helper is compiled from current runtime source. These CPU checks cover host
configuration control flow; device execution is covered by CUDA workflow tests.
"""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


_DRIVER = r'''
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
using U32=unsigned int;
using U64=unsigned long long;
using cudaError_t=int;
constexpr int cudaSuccess=0;
constexpr int cudaDevAttrMaxSharedMemoryPerBlockOptin=1;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize=2;
struct cudaFuncAttributes {std::size_t sharedSizeBytes; int maxDynamicSharedSizeBytes;};
void SHARED_KERNEL() {}
#define CUDA_TRY(expression) do {cudaError_t error=(expression); if(error!=cudaSuccess)return error;} while(false)
struct Workspace {
  U64 id; int dev;
  U64 context_id()const{return id;}
  int device()const{return dev;}
};
struct Policy {
  std::size_t statics=2208;
  int optin=101376, dynamic=46944;
  int get_fail=0, device_fail=0, set_fail=0;
  int gets=0, devices=0, sets=0;
};
std::mutex fake_mutex;
std::map<U64,Policy> policies;
thread_local U64 active_id=0;
thread_local int active_device=-1;
void require(bool value,const char*label){if(!value)throw std::runtime_error(label);}
void enter(const Workspace&w){active_id=w.id;active_device=w.dev;}
void policy(U64 id,Policy p){std::lock_guard<std::mutex> l(fake_mutex);policies[id]=p;}
Policy inspect(U64 id){std::lock_guard<std::mutex> l(fake_mutex);return policies.at(id);}
int cudaFuncGetAttributes(cudaFuncAttributes* attrs,void(*)()){
  std::lock_guard<std::mutex> l(fake_mutex);
  auto&p=policies.at(active_id);++p.gets;
  if(p.get_fail){--p.get_fail;return 30;}
  *attrs={p.statics,p.dynamic};return 0;
}
int cudaDeviceGetAttribute(int*limit,int attribute,int device){
  require(device==active_device&&attribute==cudaDevAttrMaxSharedMemoryPerBlockOptin,"device context mismatch");
  std::lock_guard<std::mutex> l(fake_mutex);
  auto&p=policies.at(active_id);++p.devices;
  if(p.device_fail){--p.device_fail;return 31;}
  *limit=p.optin;return 0;
}
int cudaFuncSetAttribute(void(*)(),int attribute,int bytes){
  require(attribute==cudaFuncAttributeMaxDynamicSharedMemorySize,"wrong attribute");
  std::lock_guard<std::mutex> l(fake_mutex);
  auto&p=policies.at(active_id);++p.sets;
  if(p.set_fail){--p.set_fail;return 32;}
  require(bytes>=0&&std::size_t(bytes)+p.statics<=std::size_t(p.optin),"resource bound exceeded");
  p.dynamic=bytes;return 0;
}

CONFIGURATION_SOURCE

void call(const Workspace& w,U32 wanted,int wanted_status=0) {
  enter(w);U32 bytes=987654;
  const int status=SmallSharedLimit(w,&bytes);
  require(status==wanted_status,"unexpected CUDA status");
  require(status ? bytes==987654 : bytes==wanted,"incorrect/premature result");
}
int main(int argc,char**argv) {
  require(argc==2,"test case required");const std::string test=argv[1];
  if(test=="capacity") {
    policy(1,{});for(int i=0;i<20;++i)call({1,0},65536);
    auto high=inspect(1);require(high.gets==1&&high.devices==1&&high.sets==1,"repeated opt-in setup");
    Policy low;low.optin=49152;policy(2,low);call({2,1},46944);
    require(inspect(2).sets==0,"unnecessary default allowance change");
    Policy exact;exact.optin=65536;policy(3,exact);call({3,0},63328);
    Policy absent;absent.optin=0;absent.dynamic=0;policy(4,absent);call({4,2},0);
    Policy static_only;static_only.optin=1024;policy(5,static_only);call({5,3},0);
    Policy large;large.optin=167936;policy(6,large);call({6,4},65536);
  } else if(test=="identity") {
    policy(1,{});call({1,0},65536);policy(2,{});call({2,0},65536);
    require(inspect(2).sets==1,"new context reused old configuration");
    Policy low;low.optin=49152;policy(3,low);call({3,1},46944);
    call({1,0},65536);require(inspect(1).sets==1,"context revisit repeated setup");
    // Deliberately repeated mock ID proves device is also part of the key.
    policy(1,{});call({1,7},65536);require(inspect(1).sets==1,"device key omitted");
  } else if(test=="retry") {
    Policy get;get.get_fail=1;policy(1,get);call({1,0},0,30);call({1,0},65536);
    require(inspect(1).gets==2&&inspect(1).sets==1,"attribute query failure cached");
    Policy device;device.device_fail=1;policy(2,device);call({2,1},0,31);call({2,1},65536);
    require(inspect(2).gets==2&&inspect(2).sets==1,"device query failure cached");
    Policy set;set.set_fail=1;policy(3,set);call({3,0},0,32);call({3,0},65536);
    require(inspect(3).sets==2,"configuration failure cached");
  } else if(test=="concurrency") {
    policy(1,{});std::vector<std::thread> threads;
    std::vector<int> errors(24,-1);std::vector<U32> limits(24,0);
    for(int i=0;i<24;++i)threads.emplace_back([&,i]{Workspace w{1,0};enter(w);errors[i]=SmallSharedLimit(w,&limits[i]);});
    for(auto& thread:threads)thread.join();
    for(int i=0;i<24;++i)require(errors[i]==0&&limits[i]==65536,"concurrent setup result differs");
    auto count=inspect(1);require(count.gets==1&&count.devices==1&&count.sets==1,"concurrent duplicate setup");
  } else throw std::runtime_error("unknown test case");
  std::cout<<test<<" passed\n";
}
'''


@pytest.fixture(scope="module")
def shared_configuration_helper(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for native configuration checks")
    path = Path(__file__).resolve().parents[1] / "src/cuda_zlib/native/codec_ffi.cu"
    source = path.read_text()
    helper = source[source.index("struct SmallSharedConfiguration {"):
                    source.index("cudaError_t Decompress(")]
    kernel = re.search(r"cudaFuncGetAttributes\(&attributes, (\w+)\)", helper).group(1)
    directory = tmp_path_factory.mktemp("small-shared-configuration")
    cpp, executable = directory / "configuration.cpp", directory / "configuration"
    cpp.write_text(_DRIVER.replace("SHARED_KERNEL", kernel)
                   .replace("CONFIGURATION_SOURCE", helper))
    built = subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", str(cpp),
                            "-o", str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    return executable


@pytest.mark.parametrize("case", ["capacity", "identity", "retry", "concurrency"])
def test_actual_small_shared_configuration(shared_configuration_helper, case):
    result = subprocess.run([str(shared_configuration_helper), case],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
