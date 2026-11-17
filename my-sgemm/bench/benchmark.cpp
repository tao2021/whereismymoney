#include "sgemm.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void check(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(e));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Event {
    cudaEvent_t e;
    Event() { check(cudaEventCreate(&e)); }
    ~Event() { cudaEventDestroy(e); }
};
struct Blas {
    cublasHandle_t h;
    Blas() {
        check(cublasCreate(&h));
        auto status = cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH);
        if (status != CUBLAS_STATUS_SUCCESS) { cublasDestroy(h); check(status); }
    }
    ~Blas() { cublasDestroy(h); }
};
float measure(const std::function<void()>& launch, int warmup, int repeat) {
    for (int i=0;i<warmup;++i) launch();
    check(cudaDeviceSynchronize());
    Event start, stop;
    check(cudaEventRecord(start.e));
    for (int i=0;i<repeat;++i) launch();
    check(cudaEventRecord(stop.e));
    check(cudaEventSynchronize(stop.e));
    float ms;
    check(cudaEventElapsedTime(&ms,start.e,stop.e));
    return ms/repeat;
}
struct Error { size_t failed=0; double max_abs=0, relative_l2=0; };
Error compare(const std::vector<float>& got, const std::vector<float>& ref, float atol, float rtol) {
    Error e;
    double diff2=0, ref2=0;
    for (size_t i=0;i<ref.size();++i) {
        double d=std::abs(double(got[i])-ref[i]);
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]) || d>atol+rtol*std::abs(ref[i])) ++e.failed;
        e.max_abs=std::max(e.max_abs,std::isfinite(d)?d:INFINITY);
        diff2+=d*d; ref2+=double(ref[i])*ref[i];
    }
    e.relative_l2=std::sqrt(diff2/std::max(ref2,1e-30));
    return e;
}
int main(int argc, char** argv) {
    try {
        int m=1024,n=1024,k=1024,warmup=5,repeat=20;
        std::string kernel="all", csv;
        for (int i=1;i<argc;++i) {
            std::string key=argv[i];
            if (key=="--help") {
                std::cout << "sgemm_benchmark [--m M --n N --k K] [--warmup W --repeat R]\n"
                             "  [--kernel all|naive|tiled|tensor_core|cublas] [--csv results.csv]\n";
                return 0;
            }
            if (i+1==argc) throw std::runtime_error("Missing value for " + key);
            std::string value=argv[++i];
            if (key=="--kernel") { kernel=value; continue; }
            if (key=="--csv") { csv=value; continue; }
            size_t used=0;
            int v=std::stoi(value,&used);
            if (used!=value.size() || v<=0 || v>1000000) throw std::runtime_error("Expected integer in [1,1000000]");
            if (key=="--m") m=v;
            else if (key=="--n") n=v;
            else if (key=="--k") k=v;
            else if (key=="--warmup") warmup=v;
            else if (key=="--repeat") repeat=v;
            else throw std::runtime_error("Unknown option " + key);
        }
        if (kernel!="all" && kernel!="naive" && kernel!="tiled" && kernel!="tensor_core" && kernel!="cublas")
            throw std::runtime_error("Unknown kernel " + kernel);
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop,0));
        check(cudaSetDevice(0));
        if ((n+15)/16>prop.maxGridSize[0] || (m+15)/16>prop.maxGridSize[1])
            throw std::runtime_error("Dimensions exceed device grid limits");
        bool tensor=prop.major>=7;
        if (!tensor && kernel=="tensor_core") throw std::runtime_error("Tensor Core requires sm_70+");
        std::cout << "GPU: " << prop.name << " sm_" << prop.major << prop.minor
                  << " | M=" << m << " N=" << n << " K=" << k
                  << " warmup=" << warmup << " repeat=" << repeat << " seed=42\n";
        size_t na=size_t(m)*k, nb=size_t(k)*n, nc=size_t(m)*n;
        std::vector<float> a(na),b(nb),ref(nc),half_ref(nc),got(nc);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1,1);
        for (auto& v:a) v=dist(rng);
        for (auto& v:b) v=dist(rng);
        Buffer<float> da(na),db(nb),dc(nc);
        check(cudaMemcpy(da.p,a.data(),na*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(db.p,b.data(),nb*sizeof(float),cudaMemcpyHostToDevice));
        Blas blas;
        float one=1,zero=0;
        auto reference=[&] {
            // Row-major C = A*B becomes column-major C^T = B^T*A^T.
            check(cublasSgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&one,db.p,n,da.p,k,&zero,dc.p,n));
        };
        reference();
        check(cudaMemcpy(ref.data(),dc.p,nc*sizeof(float),cudaMemcpyDeviceToHost));
        // Independent CPU double oracle for small test cases, including rectangular layout.
        if (double(m)*n*k<=2000000) {
            std::vector<float> cpu(nc);
            for (int r=0;r<m;++r) for (int c=0;c<n;++c) {
                double sum=0;
                for (int p=0;p<k;++p) sum+=double(a[size_t(r)*k+p])*b[size_t(p)*n+c];
                cpu[size_t(r)*n+c]=float(sum);
            }
            if (compare(ref,cpu,2e-4f,2e-4f).failed) throw std::runtime_error("cuBLAS disagrees with CPU oracle");
            std::cout << "CPU double oracle: PASS\n";
        }
        if (tensor && (kernel=="all" || kernel=="tensor_core")) {
            // Quantized-input reference isolates WMMA correctness from FP16 rounding loss.
            std::vector<float> qa=a,qb=b;
            for (auto& v:qa) v=__half2float(__float2half_rn(v));
            for (auto& v:qb) v=__half2float(__float2half_rn(v));
            check(cudaMemcpy(da.p,qa.data(),na*sizeof(float),cudaMemcpyHostToDevice));
            check(cudaMemcpy(db.p,qb.data(),nb*sizeof(float),cudaMemcpyHostToDevice));
            reference();
            check(cudaMemcpy(half_ref.data(),dc.p,nc*sizeof(float),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(da.p,a.data(),na*sizeof(float),cudaMemcpyHostToDevice));
            check(cudaMemcpy(db.p,b.data(),nb*sizeof(float),cudaMemcpyHostToDevice));
        }
        float baseline=measure(reference,warmup,repeat);
        std::ofstream file;
        if (!csv.empty()) { file.open(csv); if (!file) throw std::runtime_error("Cannot open CSV " + csv); }
        std::string header="kernel,ms,gflops,speedup_vs_cublas,max_abs_vs_fp32,relative_l2_vs_fp32,failed_elements";
        std::cout << header << '\n';
        if (file) file << header << '\n';
        bool passed=true;
        auto report=[&](const std::string& name,float ms,const Error& precision,const Error& validation) {
            auto write=[&](std::ostream& out) {
                out << std::setprecision(8) << name << ',' << ms << ',' << 2.0*m*n*k/(ms*1e6)
                    << ',' << baseline/ms << ',' << precision.max_abs << ',' << precision.relative_l2
                    << ',' << validation.failed << '\n';
            };
            write(std::cout); if(file) write(file);
            passed=passed && validation.failed==0;
        };
        report("cublas_fp32",baseline,Error{},Error{});
        struct Entry { const char* name; Launcher launch; };
        Entry entries[]={{"naive",launch_naive},{"tiled",launch_tiled},{"tensor_core",launch_tensor_core}};
        for (auto entry:entries) {
            if (kernel!="all" && kernel!=entry.name) continue;
            bool tc=std::string(entry.name)=="tensor_core";
            if (tc && !tensor) { std::cout << "SKIP tensor_core: requires sm_70+\n"; continue; }
            auto run=[&] { entry.launch(da.p,db.p,dc.p,m,n,k,0); check(cudaGetLastError()); };
            float ms=measure(run,warmup,repeat);
            // Poison output so missing stores cannot inherit a previous correct result.
            check(cudaMemset(dc.p,0xff,nc*sizeof(float)));
            run();
            check(cudaMemcpy(got.data(),dc.p,nc*sizeof(float),cudaMemcpyDeviceToHost));
            float atol=2e-5f*k, rtol=2e-4f;
            auto validation=compare(got,tc?half_ref:ref,atol,rtol);
            report(entry.name,ms,compare(got,ref,atol,rtol),validation);
        }
        if (file) { file.flush(); if (!file) throw std::runtime_error("CSV write failed"); }
        std::cout << (passed?"Validation: PASS\n":"Validation: FAIL\n");
        return passed?0:1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
