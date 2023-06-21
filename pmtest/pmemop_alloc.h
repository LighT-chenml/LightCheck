#ifndef PMEMOP_ALLOC_H
#define PMEMOP_ALLOC_H
#ifdef __cplusplus
extern "C" {
#endif

#define PM_PyObject_GC_New(type, typeobj) \
                ( (type *) _PM_PyObject_GC_New(typeobj) )

#define PM_PyObject_GC_NewVar(type, typeobj, n) \
                ( (type *) _PyObject_GC_NewVar((typeobj), (n)) )

#define PM_PyMem_New(type, n) \
  ( ((size_t)(n) > PY_SSIZE_T_MAX / sizeof(type)) ? NULL :      \
        ( (type *) PM_PyObject_Malloc((n) * sizeof(type)) ) )

PyObject *_PM_PyObject_GC_New(PyTypeObject *tp);
PyVarObject *_PyObject_GC_NewVar(PyTypeObject *tp, Py_ssize_t nitems);
void *PM_PyObject_Malloc(size_t size);
void *PM_PyObject_Malloc_Gap(size_t size);
void *PM_PyObject_Calloc(size_t nelem, size_t elsize);
void change_alloc_flag();
void set_dram_alloc();
void unset_dram_alloc();
void PM_Map();
void PM_Unmap();

#ifdef __cplusplus
}
#endif
#endif /* !PMEMOP_ALLOC_H */