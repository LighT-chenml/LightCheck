#include <Python.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/mman.h>
// #include <errno.h>
#include <cerrno>
#include <stdexcept>
#include <sys/mman.h>
#include <libpmem.h>
#include <netdb.h>
#include <string>
#include <cuda_runtime.h>
#include "pmemop_alloc.h"

#define FROM_GC(g) ((PyObject *)(((PyGC_Head *)g)+1))

size_t register_size = (size_t) 10 * 1024 * 1024 * 1024;
std::string pmemfile1 = "/mnt/pmem0/chk1";
std::string pmemfile2 = "/mnt/pmem0/chk2";
char *pmemaddr1, *pmembase1, *pmemgap_addr1;
char *pmemaddr2, *pmembase2, *pmemgap_addr2;
int aflag = 0;
int use_dram_alloc = 0;

void PM_Map()
{
    int is_pmem;
    size_t mapped_len;
    pmembase1 = (char *)pmem_map_file(pmemfile1.c_str(), register_size, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem);
    if (pmembase1 == NULL) {
        printf("pmem_map_file fail\n");
        exit(1);
    }
    auto cuda_retval = cudaHostRegister (pmembase1, register_size, cudaHostRegisterMapped | cudaHostRegisterPortable);
	if (cuda_retval != cudaSuccess)
		throw std::runtime_error (cudaGetErrorString (cuda_retval) + std::string (" (cudaHostRegister)"));

    pmembase2 = (char *)pmem_map_file(pmemfile2.c_str(), register_size, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem);
    if (pmembase2 == NULL) {
        printf("pmem_map_file fail\n");
        exit(1);
    }
    cuda_retval = cudaHostRegister (pmembase2, register_size, cudaHostRegisterMapped | cudaHostRegisterPortable);
	if (cuda_retval != cudaSuccess)
		throw std::runtime_error (cudaGetErrorString (cuda_retval) + std::string (" (cudaHostRegister)"));

    pmemaddr1 = pmembase1;
    pmemgap_addr1 = pmembase1 + ((size_t) 1 * 1024 * 1024 * 1024);
    pmemaddr2 = pmembase2;
    pmemgap_addr2 = pmembase2 + ((size_t) 1 * 1024 * 1024 * 1024);
    aflag = 0;
    printf("map pm file\n");
}

void PM_Unmap()
{
    cudaHostUnregister (pmembase1);
    pmem_unmap(pmembase1, register_size);
    cudaHostUnregister (pmembase2);
    pmem_unmap(pmembase2, register_size);
    printf("unmap pm file\n");
}

void change_alloc_flag() {
    aflag = (aflag == 0 ? 1 : 0);
}

void set_dram_alloc() {
    use_dram_alloc = 1;
}

void unset_dram_alloc() {
    use_dram_alloc = 0;
}

static PyObject *pm_gc_alloc(size_t basicsize)
{
    PyObject *op;
    PyGC_Head *g;
    size_t size = sizeof(PyGC_Head) + basicsize;
    g = (PyGC_Head *)PM_PyObject_Malloc(size);
    if (g == NULL)
        return NULL;
    g->gc.gc_refs = 0;
    op = FROM_GC(g);
    return op;
}

PyObject *_PM_PyObject_GC_New(PyTypeObject *tp)
{
    PyObject *op = pm_gc_alloc(_PyObject_SIZE(tp));
    if (op == NULL) {
        return NULL;
    }
    op = PyObject_INIT(op, tp);
    return op;
}

PyVarObject *_PyObject_GC_NewVar(PyTypeObject *tp, Py_ssize_t nitems)
{
    size_t size;
    PyVarObject *op;
    if (nitems < 0) {
        PyErr_BadInternalCall();
        return NULL;
    }
    size = _PyObject_VAR_SIZE(tp, nitems);
    op = (PyVarObject *)pm_gc_alloc(size);
    if (op == NULL) {
        return NULL;
    }
    op = PyObject_InitVar(op, tp, nitems);
    return op;
}

void *PM_PyObject_Malloc(size_t size)
{
    if (size == 0)
        size = 1;
    if (use_dram_alloc)
        return malloc(size);
    char *alloc_addr;
    if (aflag == 0) {
        alloc_addr = pmemaddr1;
        if (pmemaddr1 + size > pmembase1 + register_size) {
            printf("PM space is not enough!\n");
            exit(1);
        }
        pmemaddr1 += size;
        pmemaddr1 = (char *)(((uint64_t)(pmemaddr1 + 255)) & (0xffffffffff00));
    }
    else {
        alloc_addr = pmemaddr2;
        if (pmemaddr2 + size > pmembase2 + register_size) {
            printf("PM space is not enough!\n");
            exit(1);
        }
        pmemaddr2 += size;
        pmemaddr2 = (char *)(((uint64_t)(pmemaddr2 + 255)) & (0xffffffffff00));
    }
    return alloc_addr;
}

void *PM_PyObject_Malloc_Gap(size_t size)
{
    if (size == 0)
        size = 1;
    if (use_dram_alloc)
        return malloc(size);
    char *alloc_addr;
    if (aflag == 0) {
        alloc_addr = pmemgap_addr1;
        if (pmemgap_addr1 + size > pmembase1 + register_size) {
            printf("PM space is not enough!\n");
            exit(1);
        }
        pmemgap_addr1 += size;
        pmemgap_addr1 = (char *)(((uint64_t)(pmemgap_addr1 + 255)) & (0xffffffffff00));
    }
    else {
        alloc_addr = pmemgap_addr2;
        if (pmemgap_addr2 + size > pmembase2 + register_size) {
            printf("PM space is not enough!\n");
            exit(1);
        }
        pmemgap_addr2 += size;
        pmemgap_addr2 = (char *)(((uint64_t)(pmemgap_addr2 + 255)) & (0xffffffffff00));
    }
    // printf("alloc addr: %p\n", alloc_addr);
    return alloc_addr;
}

void *PM_PyObject_Calloc(size_t nelem, size_t elsize)
{
    if (nelem == 0 || elsize == 0) {
        nelem = 1;
        elsize = 1;
    }
    if (use_dram_alloc)
        return calloc(nelem, elsize);
    char *alloc_addr;
    if (aflag == 0) {
        alloc_addr = pmemaddr1;
        if (pmemaddr1 + nelem * elsize > pmembase1 + register_size) {
            printf("PM space is not enough!\n");
            exit(1);
        }
        pmemaddr1 += nelem * elsize;
        pmemaddr1 = (char *)(((uint64_t)(pmemaddr1 + 255)) & (0xffffffffff00));
        memset(alloc_addr, 0, nelem * elsize);
    }
    else {
        alloc_addr = pmemaddr2;
        if (pmemaddr2 + nelem * elsize > pmembase2 + register_size) {
            printf("PM space is not enough!\n");
            exit(1);
        }
        pmemaddr2 += nelem * elsize;
        pmemaddr2 = (char *)(((uint64_t)(pmemaddr2 + 255)) & (0xffffffffff00));
        memset(alloc_addr, 0, nelem * elsize);
    }
    return alloc_addr;
}
