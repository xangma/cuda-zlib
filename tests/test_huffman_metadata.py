# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Compare the actual small cached Huffman helper with the canonical decoder."""

import ast
import os
from pathlib import Path
import shutil
import subprocess

import pytest

_DRIVER = r'''
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#define __device__
#define __forceinline__ inline
using U8=unsigned char;
using U32=unsigned int;
using U64=unsigned long long;
void require(bool value,const char*message){if(!value){std::cerr<<message<<'\n';std::exit(1);}}

namespace decoder {
PARSER_SOURCE
}
namespace candidate {
SMALL_HELPER_SOURCE
}
bool same(const decoder::BitReader&a,const decoder::BitReader&b){
 return a.data==b.data&&a.bits==b.bits&&a.pos==b.pos&&a.error==b.error&&a.cache==b.cache&&a.cached==b.cached;
}
template<int N,int P> U64 check_table(decoder::Huffman<N,P>&table,U8*end){
 U64 checked=0;
 for(U64 bits=0;bits<=80;++bits){
  U8*data=end-(bits+7)/8;
  for(U64 i=0;i<(bits+7)/8;++i)data[i]=U8(0x73+37*i+bits);
  for(U64 start=0;start<=bits+1;++start)for(U32 seeded=0;seeded<2;++seeded)for(U32 err: {0u,1u,4u}){
   decoder::BitReader a{data,bits,start,0,0,0};
   if(seeded&&start<=bits)a.peek(U32(std::min<U64>(bits-start,16)));
   a.error=err;auto b=a;
   int old=table.decode(a),newer=candidate::SmallHuffmanDecode(b,table,table.maximum,table.lookup);
   require(old==newer&&same(a,b),"Huffman result/reader state mismatch");++checked;
  }
 }
 // Every possible low-byte prefix, full primary lookup and long-code fallback.
 for(U32 byte=0;byte<256;++byte){
  U8*data=end-4;data[0]=U8(byte);data[1]=U8(byte^0xa5);data[2]=0x37;data[3]=0xfb;
  for(U64 start=0;start<8;++start){
   decoder::BitReader a{data,32,start,0,0,0};auto b=a;
   int old=table.decode(a),newer=candidate::SmallHuffmanDecode(b,table,table.maximum,table.lookup);
   require(old==newer&&same(a,b),"prefix mismatch");++checked;
  }
 }
 return checked;
}
void helper_checks(){
 const long page=sysconf(_SC_PAGESIZE);
 void*mapping=mmap(nullptr,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 require(mapping!=MAP_FAILED&&!mprotect(static_cast<U8*>(mapping)+page,page,PROT_NONE),"mmap");
 U8*end=static_cast<U8*>(mapping)+page;U64 cases=0;
 decoder::DecodeTables tables{};require(decoder::fixed_tables(tables.ll,tables.dd)==0,"fixed table build");
 cases+=check_table(tables.ll,end);cases+=check_table(tables.dd,end);
 std::array<decoder::u8,288> lens{};
 for(U32 i=0;i<14;++i)lens[i]=decoder::u8(i+1);lens[14]=lens[15]=15;
 for(bool lookup:{false,true}){
  require(tables.ll.build(lens.data(),288,0,0,15,lookup)==0,"long table build");cases+=check_table(tables.ll,end);
  require(tables.dd.build(lens.data(),32,0,0,15,lookup)==0,"long distance build");cases+=check_table(tables.dd,end);
 }
 lens.fill(0);require(tables.dd.build(lens.data(),1,1,1,15)==0,"empty distance build");cases+=check_table(tables.dd,end);
 lens[256]=1;require(tables.ll.build(lens.data(),257,0,1,15)==0,"single EOB build");cases+=check_table(tables.ll,end);
 lens.fill(0);lens[0]=1;require(tables.dd.build(lens.data(),1,1,1,15)==0,"single distance build");cases+=check_table(tables.dd,end);
 require(munmap(mapping,page*2)==0,"munmap");std::cout<<cases<<"\n";
}

int main(){helper_checks();}
'''


def test_small_huffman_metadata_matches_canonical(tmp_path):
    if os.name != "posix":
        pytest.skip("guard-page checks require POSIX mmap")
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    root = Path(__file__).resolve().parents[1]
    module = root / "src/cuda_zlib/_decode_kernels.py"
    cuda = next(ast.literal_eval(node.value) for node in ast.parse(module.read_text()).body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "CUDA_SOURCE"
                        for target in node.targets))
    parser = cuda.split("// Each scan thread loads", 1)[0]
    header = (root / "src/cuda_zlib/native/batch_decode.cuh").read_text()
    begin = header.index("template<int N, int PrimaryBits>\n__device__ __forceinline__ int SmallHuffmanDecode(")
    helper = header[begin:header.index("// Small/batch token decoding", begin)]
    source = tmp_path / "huffman_metadata.cpp"
    source.write_text(_DRIVER.replace("PARSER_SOURCE", parser).replace("SMALL_HELPER_SOURCE", helper))
    binary = tmp_path / "huffman_metadata"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert int(result.stdout) > 0
