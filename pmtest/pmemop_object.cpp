#include <Python.h>
#include <stddef.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <iostream>
#include <libpmem.h>
#include "pmemop_object.h"
#include "pmemop_alloc.h"

ChkRequest::ChkRequest(int req_type, PyObject *dict, PyObject *key, PyObject *value, int handle)
    : req_type(req_type),
      dict(dict),
      key(key),
      value(value),
      handle(handle)
{

}

static Py_ssize_t lookdict_split(PyDictObject *mp, PyObject *key,
                                 Py_hash_t hash, PyObject ***value_addr,
                                 Py_ssize_t *hashpos);

static PyDictKeysObject empty_keys_struct = {
        1, /* dk_refcnt */
        1, /* dk_size */
        lookdict_split, /* dk_lookup */
        0, /* dk_usable (immutable) */
        0, /* dk_nentries */
        {DKIX_EMPTY, DKIX_EMPTY, DKIX_EMPTY, DKIX_EMPTY,
         DKIX_EMPTY, DKIX_EMPTY, DKIX_EMPTY, DKIX_EMPTY}, /* dk_indices */
};

static PyObject *empty_values[1] = { NULL };
static long chk_way = 0;

void set_chk_way(long way) {
    chk_way = chk_way;
}

#define Py_EMPTY_KEYS &empty_keys_struct

#define PERTURB_SHIFT 5

#define DK_SIZE(dk) ((dk)->dk_size)
#if SIZEOF_VOID_P > 4
#define DK_IXSIZE(dk)                          \
    (DK_SIZE(dk) <= 0xff ?                     \
        1 : DK_SIZE(dk) <= 0xffff ?            \
            2 : DK_SIZE(dk) <= 0xffffffff ?    \
                4 : sizeof(int64_t))
#else
#define DK_IXSIZE(dk)                          \
    (DK_SIZE(dk) <= 0xff ?                     \
        1 : DK_SIZE(dk) <= 0xffff ?            \
            2 : sizeof(int32_t))
#endif
#define DK_ENTRIES(dk) \
    ((PyDictKeyEntry*)(&((int8_t*)((dk)->dk_indices))[DK_SIZE(dk) * DK_IXSIZE(dk)]))

#define DK_MASK(dk) (((dk)->dk_size)-1)

#define pm_new_values(size) PM_PyMem_New(PyObject *, size)

static inline void dictkeys_incref(PyDictKeysObject *dk)
{
    dk->dk_refcnt++;
}

static inline void dictkeys_decref(PyDictKeysObject *dk)
{
    dk->dk_refcnt--;
}

Py_LOCAL_INLINE(int)
unicode_eq(PyObject *aa, PyObject *bb)
{
    PyUnicodeObject *a = (PyUnicodeObject *)aa;
    PyUnicodeObject *b = (PyUnicodeObject *)bb;

    if (PyUnicode_READY(a) == -1 || PyUnicode_READY(b) == -1) {
        assert(0 && "unicode_eq ready fail");
        return 0;
    }

    if (PyUnicode_GET_LENGTH(a) != PyUnicode_GET_LENGTH(b))
        return 0;
    if (PyUnicode_GET_LENGTH(a) == 0)
        return 1;
    if (PyUnicode_KIND(a) != PyUnicode_KIND(b))
        return 0;
    return memcmp(PyUnicode_1BYTE_DATA(a), PyUnicode_1BYTE_DATA(b),
                  PyUnicode_GET_LENGTH(a) * PyUnicode_KIND(a)) == 0;
}

static inline Py_ssize_t dk_get_index(PyDictKeysObject *keys, Py_ssize_t i)
{
    Py_ssize_t s = DK_SIZE(keys);
    Py_ssize_t ix;

    if (s <= 0xff) {
        int8_t *indices = (int8_t*)(keys->dk_indices);
        ix = indices[i];
    }
    else if (s <= 0xffff) {
        int16_t *indices = (int16_t*)(keys->dk_indices);
        ix = indices[i];
    }
#if SIZEOF_VOID_P > 4
    else if (s > 0xffffffff) {
        int64_t *indices = (int64_t*)(keys->dk_indices);
        ix = indices[i];
    }
#endif
    else {
        int32_t *indices = (int32_t*)(keys->dk_indices);
        ix = indices[i];
    }
    assert(ix >= DKIX_DUMMY);
    return ix;
}

static Py_ssize_t
lookdict_split(PyDictObject *mp, PyObject *key,
               Py_hash_t hash, PyObject ***value_addr, Py_ssize_t *hashpos)
{
    return 0;
}

static PyDictObject *pm_new_dict(PyDictKeysObject *keys, PyObject **values, Py_ssize_t used, int free_values_on_failure)
{
    PyDictObject *mp;
    assert(keys != NULL);
    mp = PM_PyObject_GC_New(PyDictObject, &PyDict_Type);
    if (mp == NULL) {
        return NULL;
    }
    mp->ma_keys = keys;
    mp->ma_values = values;
    mp->ma_used = used;
    mp->ma_version_tag = 0;
    return mp;
}

static inline Py_ssize_t shared_keys_usable_size(PyDictKeysObject *keys)
{
    return keys->dk_nentries + keys->dk_usable;
}

dict_lookup_func lookdict;

static Py_ssize_t lookdict_unicode(PyDictObject *mp, PyObject *key,
                 Py_hash_t hash, PyObject ***value_addr, Py_ssize_t *hashpos)
{
    // printf("go into lookdict_unicode\n");
    size_t i;
    size_t mask = DK_MASK(mp->ma_keys);
    Py_ssize_t ix, freeslot;
    PyDictKeyEntry *ep, *ep0 = DK_ENTRIES(mp->ma_keys);

    assert(mp->ma_values == NULL);

    if (!PyUnicode_CheckExact(key)) {
        mp->ma_keys->dk_lookup = lookdict;
        return lookdict(mp, key, hash, value_addr, hashpos);
    }

    i = (size_t)hash & mask;
    ix = dk_get_index(mp->ma_keys, i);

    if (ix == DKIX_EMPTY) {
        if (hashpos != NULL)
            *hashpos = i;
        *value_addr = NULL;
        return DKIX_EMPTY;
    }

    if (ix == DKIX_DUMMY) {
        freeslot = i;
    }
    else {
        ep = &ep0[ix];
        assert(ep->me_key != NULL);
        // printf("me_key: %s, key, %s\n", PyUnicode_AS_DATA(ep->me_key), PyUnicode_AS_DATA(key));
        // printf("me_key: %s, key, %s\n", PyUnicode_AsUTF8(ep->me_key), PyUnicode_AsUTF8(key));
        // printf("me_hash: %ld, hash: %ld\n", ep->me_hash, hash);
        // if (unicode_eq(ep->me_key, key))
        //     printf("equal\n");
        // else
        //     printf("not equal\n");
        if (ep->me_key == key
            || (ep->me_hash == hash && unicode_eq(ep->me_key, key))) {
            if (hashpos != NULL)
                *hashpos = i;
            *value_addr = &ep->me_value;
            return ix;
        }
        freeslot = -1;
    }

    for (size_t perturb = hash;;) {
        perturb >>= PERTURB_SHIFT;
        i = mask & ((i << 2) + i + perturb + 1);
        ix = dk_get_index(mp->ma_keys, i);
        if (ix == DKIX_EMPTY) {
            if (hashpos != NULL) {
                *hashpos = (freeslot == -1) ? (Py_ssize_t)i : freeslot;
            }
            *value_addr = NULL;
            return DKIX_EMPTY;
        }
        if (ix == DKIX_DUMMY) {
            if (freeslot == -1)
                freeslot = i;
            continue;
        }
        ep = &ep0[ix];
        assert(ep->me_key != NULL);
        if (ep->me_key == key
            || (ep->me_hash == hash && unicode_eq(ep->me_key, key))) {
            *value_addr = &ep->me_value;
            if (hashpos != NULL) {
                *hashpos = i;
            }
            return ix;
        }
    }
    assert(0);          /* NOT REACHED */
    return 0;
}

static Py_ssize_t comm_lookdict(PyDictObject *mp, PyObject *key,
         Py_hash_t hash, PyObject ***value_addr, Py_ssize_t *hashpos)
{
    // printf("go into common lookdict\n");
    size_t i, mask;
    Py_ssize_t ix, freeslot;
    PyDictKeysObject *dk;
    PyDictKeyEntry *ep0, *ep;

    dk = mp->ma_keys;
    mask = DK_MASK(dk);
    ep0 = DK_ENTRIES(dk);
    i = (size_t)hash & mask;

    ix = dk_get_index(dk, i);
    if (ix == DKIX_EMPTY) {
        if (hashpos != NULL)
            *hashpos = i;
        *value_addr = NULL;
        return DKIX_EMPTY;
    }
    if (ix == DKIX_DUMMY) {
        freeslot = i;
    }
    else {
        ep = &ep0[ix];
        assert(ep->me_key != NULL);
        if (ep->me_key == key) {
            *value_addr = &ep->me_value;
            if (hashpos != NULL)
                *hashpos = i;
            return ix;
        }
        // printf("me_hash: %ld, hash: %ld\n", ep->me_hash, hash);
        if (ep->me_hash == hash) {
            *value_addr = &ep->me_value;
            if (hashpos != NULL)
                *hashpos = i;
            return ix;
        }
        freeslot = -1;
    }

    for (size_t perturb = hash;;) {
        perturb >>= PERTURB_SHIFT;
        i = ((i << 2) + i + perturb + 1) & mask;
        ix = dk_get_index(dk, i);
        if (ix == DKIX_EMPTY) {
            if (hashpos != NULL) {
                *hashpos = (freeslot == -1) ? (Py_ssize_t)i : freeslot;
            }
            *value_addr = NULL;
            return ix;
        }
        if (ix == DKIX_DUMMY) {
            if (freeslot == -1)
                freeslot = i;
            continue;
        }
        ep = &ep0[ix];
        assert(ep->me_key != NULL);
        if (ep->me_key == key) {
            if (hashpos != NULL) {
                *hashpos = i;
            }
            *value_addr = &ep->me_value;
            return ix;
        }
        if (ep->me_hash == hash) {
            if (hashpos != NULL) {
                *hashpos = i;
            }
            *value_addr = &ep->me_value;
            return ix;
        }
    }
    assert(0);          /* NOT REACHED */
    return 0;
}

static inline PyDictKeysObject *pm_new_keys_write(PyDictKeysObject *orig_keys)
{
    Py_ssize_t keys_size = All_PyDict_KeysSize(orig_keys);
    PyDictKeysObject *keys =(PyDictKeysObject *)PM_PyObject_Malloc(keys_size);
    if (keys == NULL) {
        printf("allocate keys memory failed\n");
        return NULL;
    }
    keys->dk_refcnt = 1;
    keys->dk_size = orig_keys->dk_size;
    keys->dk_lookup = lookdict_unicode;
    // lookdict = orig_keys->dk_lookup;
    lookdict = comm_lookdict;
    keys->dk_usable = orig_keys->dk_usable;
    keys->dk_nentries = orig_keys->dk_nentries;
    memcpy(keys->dk_indices, orig_keys->dk_indices, DK_SIZE(orig_keys) * DK_IXSIZE(orig_keys));
    PyDictKeyEntry *ep0 = DK_ENTRIES(keys);
    PyDictKeyEntry *orig_ep0 = DK_ENTRIES(orig_keys);
    Py_ssize_t n = orig_keys->dk_nentries;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyDictKeyEntry *entry = &ep0[i];
        PyDictKeyEntry *orig_entry = &orig_ep0[i];
        PyObject *orig_key = orig_entry->me_key;
        if (orig_key != NULL) {
            if (PyUnicode_Check(orig_key)) {
                entry->me_key = PM_PyUnicode_Write(orig_key);
            }
            else if (PyFloat_Check(orig_key)) {
                double x = PyFloat_AS_DOUBLE(orig_key);
                entry->me_key = PM_PyFloat_FromDouble_Write(x);
            }
            else if (PyLong_Check(orig_key)) {
                entry->me_key = PM_PyLong_Write(orig_key);
            }
            else {
                printf("key is other types\n");
                exit(1);
            }
            entry->me_hash = orig_entry->me_hash;
        }
    }

    return keys;
}

static inline void pm_new_keys_copy(PyDictKeysObject *dst_keys, PyDictKeysObject *orig_keys)
{
    PyDictKeyEntry *ep0 = DK_ENTRIES(dst_keys);
    PyDictKeyEntry *orig_ep0 = DK_ENTRIES(orig_keys);
    Py_ssize_t n = orig_keys->dk_nentries;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyDictKeyEntry *entry = &ep0[i];
        PyDictKeyEntry *orig_entry = &orig_ep0[i];
        PyObject *orig_key = orig_entry->me_key;
        if (orig_key != NULL) {
            if (PyUnicode_Check(orig_key)) {
                PM_PyUnicode_Copy(entry->me_key, orig_key);
            }
            else if (PyFloat_Check(orig_key)) {
                double x = PyFloat_AS_DOUBLE(orig_key);
                PM_PyFloat_FromDouble_Copy(entry->me_key, x);
            }
            else if (PyLong_Check(orig_key)) {
                PM_PyLong_Copy(entry->me_key, orig_key);
            }
            else {
                printf("key is other types\n");
                exit(1);
            }
            entry->me_hash = orig_entry->me_hash;
        }
    }
}

static PyDictKeysObject *pm_write_combined_dict_keys(PyDictObject *orig)
{
    Py_ssize_t keys_size = All_PyDict_KeysSize(orig->ma_keys);
    PyDictKeysObject *keys = (PyDictKeysObject *)PM_PyObject_Malloc(keys_size);
    if (keys == NULL) {
        printf("allocate keys memory failed\n");
        return NULL;
    }
    keys->dk_refcnt = 1;
    keys->dk_size = orig->ma_keys->dk_size;
    keys->dk_lookup = lookdict_unicode;
    // lookdict = orig->ma_keys->dk_lookup;
    lookdict = comm_lookdict;
    keys->dk_usable = orig->ma_keys->dk_usable;
    keys->dk_nentries = orig->ma_keys->dk_nentries;
    memcpy(keys->dk_indices, orig->ma_keys->dk_indices, DK_SIZE(orig->ma_keys) * DK_IXSIZE(orig->ma_keys));
    PyDictKeyEntry *ep0 = DK_ENTRIES(keys);
    PyDictKeyEntry *orig_ep0 = DK_ENTRIES(orig->ma_keys);
    Py_ssize_t n = orig->ma_keys->dk_nentries;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyDictKeyEntry *entry = &ep0[i];
        PyDictKeyEntry *orig_entry = &orig_ep0[i];
        PyObject *orig_value = orig_entry->me_value;
        if (orig_value != NULL) {
            // check whether value is tensor type
            if (THPVariableClass && PyObject_IsInstance(orig_value, THPVariableClass)) {
                entry->me_value = PM_PyTensor_Write(orig_value);
            }
            else if (PyDict_Check(orig_value)) {
                entry->me_value = PM_PyDict_Write(orig_value);
            }
            else if (PyList_Check(orig_value)) {
                entry->me_value = PM_PyList_Write(orig_value);
            }
            else if (PyUnicode_Check(orig_value)) {
                entry->me_value = PM_PyUnicode_Write(orig_value);
            }
            else if (PyFloat_Check(orig_value)) {
                double x = PyFloat_AS_DOUBLE(orig_value);
                entry->me_value = PM_PyFloat_FromDouble_Write(x);
            }
            else if (PyLong_Check(orig_value)) {
                entry->me_value = PM_PyLong_Write(orig_value);
            }
            else if (PyTuple_Check(orig_value)) {
                entry->me_value = PM_PyTuple_Write(orig_value);
            }
            else {
                printf("pm_write_combined_dict_keys error: value is other types, %s\n", Py_TYPE(orig_value)->tp_name);
                exit(1);
            }

            PyObject *orig_key = orig_entry->me_key;
            if (PyUnicode_Check(orig_key)) {
                entry->me_key = PM_PyUnicode_Write(orig_key);
            }
            else if (PyFloat_Check(orig_key)) {
                double x = PyFloat_AS_DOUBLE(orig_key);
                entry->me_key = PM_PyFloat_FromDouble_Write(x);
            }
            else if (PyLong_Check(orig_key)) {
                entry->me_key = PM_PyLong_Write(orig_key);
            }
            else {
                printf("key is other types\n");
                exit(1);
            }

            entry->me_hash = orig_entry->me_hash;
        }
    }

    return keys;
}

static void pm_copy_combined_dict_keys(ChkGlobalState& chk_state, PyDictObject *dst, PyDictObject *orig)
{
    PyDictKeyEntry *ep0 = DK_ENTRIES(dst->ma_keys);
    PyDictKeyEntry *orig_ep0 = DK_ENTRIES(orig->ma_keys);
    Py_ssize_t n = orig->ma_keys->dk_nentries;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyDictKeyEntry *entry = &ep0[i];
        PyDictKeyEntry *orig_entry = &orig_ep0[i];
        PyObject *orig_value = orig_entry->me_value;
        if (orig_value != NULL) {
            PyObject *orig_key = orig_entry->me_key;
            // if (PyUnicode_Check(orig_key)) {
            //     if (!PyUnicode_Check(entry->me_key)) {
            //         printf("the key type is not equal\n");
            //         exit(1);
            //     }
            //     if (!unicode_eq(orig_key, entry->me_key)) {
            //         printf("the corresponding key is not equal\n");
            //         exit(1);
            //     }
            //     // printf("str key: %s\n", PyUnicode_AsUTF8(entry->me_key));
            // }
            // else if (PyFloat_Check(orig_key)) {
            //     if (!PyFloat_Check(entry->me_key)) {
            //         printf("the key type is not equal\n");
            //         exit(1);
            //     }
            //     double x = PyFloat_AS_DOUBLE(orig_key);
            //     double y = PyFloat_AS_DOUBLE(entry->me_key);
            //     if (x != y) {
            //         printf("the corresponding key is not equal\n");
            //         exit(1);
            //     }
            //     // std::cout << "float key: " << x << std::endl;
            // }
            // else if (PyLong_Check(orig_key)) {
            //     if (!PyLong_Check(entry->me_key)) {
            //         printf("the key type is not equal\n");
            //         exit(1);
            //     }
            //     long x = PyLong_AsLong(orig_key);
            //     long y = PyLong_AsLong(entry->me_key);
            //     if (x != y) {
            //         printf("the corresponding key is not equal\n");
            //         exit(1);
            //     }
            //     // std::cout << "long key: " << x << std::endl;
            // }
            // else {
            //     printf("key is other types\n");
            //     exit(1);
            // }

            if (PyDict_Check(orig_value)) {
                PM_PyDict_Copy(chk_state, entry->me_value, orig_value);
            }
            else if (PyList_Check(orig_value)) {
                PM_PyList_Copy(chk_state, entry->me_value, orig_value);
            }
            else if (PyUnicode_Check(orig_value)) {
                PM_PyUnicode_Copy(entry->me_value, orig_value);
            }
            else if (PyFloat_Check(orig_value)) {
                double x = PyFloat_AS_DOUBLE(orig_value);
                PM_PyFloat_FromDouble_Copy(entry->me_value, x);
            }
            else if (PyLong_Check(orig_value)) {
                PM_PyLong_Copy(entry->me_value, orig_value);
            }
            else if (PyTuple_Check(orig_value)) {
                PM_PyTuple_Copy(chk_state, entry->me_value, orig_value);
            }
            // check whether value is tensor type
            // else if (PyObject_IsInstance(orig_value, THPVariableClass)) {
            //     PM_PyTensor_Copy(chk_state, entry->me_value, orig_value);
            // }
            else {
                // printf("value is other types\n");
                // exit(1);
                PM_PyTensor_Copy(chk_state, entry->me_value, orig_value);
            }
            // entry->me_hash = orig_entry->me_hash;
        }
    }
}

// static int pm_insertdict(PyDictObject *mp, PyObject *key, Py_hash_t hash, PyObject *value)
// {
//     printf("start pm_insertdict\n");
//     PyObject *old_value;
//     PyObject **value_addr;
//     PyDictKeyEntry *ep, *ep0;
//     Py_ssize_t hashpos, ix;

//     ix = mp->ma_keys->dk_lookup(mp, key, hash, &value_addr, &hashpos);
//     printf("ix: %ld\n",ix);
//     if (ix == DKIX_ERROR)
//         goto Fail;

//     assert(value_addr != NULL);

//     old_value = *value_addr;
//     if (old_value != NULL) {
//         if (PyDict_Check(value)) {
//             PM_PyDict_Copy(old_value, value);
//         }
//         else if (PyList_Check(value)) {
//             PM_PyList_Copy(old_value, value);
//         }
//         else if (PyUnicode_Check(value)) {
//             PM_PyUnicode_Copy(old_value, value);
//         }
//         else if (PyFloat_Check(value)) {
//             double x = PyFloat_AS_DOUBLE(value);
//             PM_PyFloat_FromDouble_Copy(old_value, x);
//         }
//         else if (PyLong_Check(value)) {
//             PM_PyLong_Copy(old_value, value);
//         }
//         else {
//             printf("value is other types\n");
//             exit(1);
//         }
//         mp->ma_version_tag = 0;
//         return 0;
//     }
//     else {
//         printf("old value is null\n");
//     }

// Fail:
//     return -1;
// }

void ChkThreadLoop(ChkGlobalState& chk_state)
{
    cpu_set_t cpu_set;
    pthread_t current_thread = pthread_self();
    int affinity = 40;
    CPU_ZERO(&cpu_set);
    CPU_SET(affinity, &cpu_set);
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpu_set) != 0) {
        printf("set affinity failed");
    }
    chk_state.initial_done = true;
    while (true) {
        if (RunLoop(chk_state) == false)
            break;
    }
}

bool RunLoop(ChkGlobalState& chk_state)
{
    std::unique_lock<std::mutex> lk(chk_state.que_mutex);
    chk_state.que_cond.wait(lk, [&chk_state] {return !chk_state.chk_que.empty();});
    // ChkRequest& req = chk_state.chk_que.front();
    // std::shared_ptr<struct ChkRequest> req = std::move(chk_state.chk_que.front());
    std::shared_ptr<struct ChkRequest> req = chk_state.chk_que.front();
    chk_state.chk_que.pop();
    lk.unlock();
    // process data
    if (req->req_type == 0) {
        struct timespec start, end;
        float imc_rd = 0, imc_wr = 0, media_rd = 0, media_wr = 0;
        // PmmDataCollector("dimm", &imc_rd, &imc_wr, &media_rd, &media_wr);
        clock_gettime(CLOCK_REALTIME, &start);
        if (chk_state.chk_flag == 0) {
            PM_PyDict_Copy(chk_state, chk_state.pm_dict[0], req->dict);
        }
        else if (chk_state.chk_flag == 1) {
            PM_PyDict_Copy(chk_state, chk_state.pm_dict[1], req->dict);
        }
        else {
            return false;
        }
        clock_gettime(CLOCK_REALTIME, &end);
        std::lock_guard<std::mutex> guard(chk_state.save_mutex);
        // std::cout << "sub thread set save_flag 1" << std::endl;
        chk_state.save_flag = 1;
        double msec = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
        printf("store chk_dict takes %.2lfms\n", msec);
        // PmmDataFinish();
    }
    else if (req->req_type == 1) {
        if (chk_state.chk_flag == 0) {
            PM_PyDict_SetItem(chk_state, chk_state.pm_dict[0], req);
        }
        else if (chk_state.chk_flag == 1) {
            PM_PyDict_SetItem(chk_state, chk_state.pm_dict[1], req);
        }
    }
    else if (req->req_type == 2) {
        if (chk_state.chk_flag == 0) {
            PM_PyDict_SetItem(chk_state, chk_state.model_dict[0], req);
        }
        else if (chk_state.chk_flag == 1) {
            PM_PyDict_SetItem(chk_state, chk_state.model_dict[1], req);
        }
    }
    else if (req->req_type == 3) {
        if (chk_state.chk_flag == 0) {
            PM_PyDict_SetItem(chk_state, chk_state.opt_dict[0], req);
        }
        else if (chk_state.chk_flag == 1) {
            PM_PyDict_SetItem(chk_state, chk_state.opt_dict[1], req);
        }
    }
    else if (req->req_type == 4) {
        if (chk_state.chk_flag == 0) {
            PM_PyDict_SetItem(chk_state, chk_state.opt_state_dict[0], req);
        }
        else if (chk_state.chk_flag == 1) {
            PM_PyDict_SetItem(chk_state, chk_state.opt_state_dict[1], req);
        }
    }
    else if (req->req_type == 5) {
        std::cout << "sub chk thread end" << std::endl;
        return false;
    }
    return true;
}

int PM_PyDict_SetItem(ChkGlobalState& chk_state, PyObject *dict, std::shared_ptr<struct ChkRequest> req)
{
    PyDictObject *mp;
    Py_hash_t hash;
    mp = (PyDictObject *)dict;
    PyObject *key = req->key;
    PyObject *value = req->value;
    if (!PyUnicode_CheckExact(key) ||
        (hash = ((PyASCIIObject *) key)->hash) == -1)
    {
        // printf("setitem1*\n");
        hash = PyObject_Hash(key);
        if (hash == -1)
            return -1;
    }
    PyObject *old_value;
    PyObject **value_addr;
    Py_ssize_t hashpos, ix;
    ix = mp->ma_keys->dk_lookup(mp, key, hash, &value_addr, &hashpos);
    if (ix == DKIX_ERROR)
        goto Fail;

    assert(value_addr != NULL);

    // if (PyUnicode_Check(key)) {
    //     printf("str key: %s\n", PyUnicode_AsUTF8(key));
    // }
    // else if (PyFloat_Check(key)) {
    //     double x = PyFloat_AS_DOUBLE(key);
    //     std::cout << "float key: " << x << std::endl;
    // }
    // else if (PyLong_Check(key)) {
    //     long x = PyLong_AsLong(key);
    //     std::cout << "long key: " << x << std::endl;
    // }

    old_value = *value_addr;
    if (old_value != NULL) {
        if (PyDict_Check(value)) {
            PM_PyDict_Copy(chk_state, old_value, value);
        }
        else if (PyList_Check(value)) {
            PM_PyList_Copy(chk_state, old_value, value);
        }
        else if (PyUnicode_Check(value)) {
            PM_PyUnicode_Copy(old_value, value);
        }
        else if (PyFloat_Check(value)) {
            double x = PyFloat_AS_DOUBLE(value);
            PM_PyFloat_FromDouble_Copy(old_value, x);
        }
        else if (PyLong_Check(value)) {
            PM_PyLong_Copy(old_value, value);
        }
        else if (PyTuple_Check(value)) {
            PM_PyTuple_Copy(chk_state, old_value, value);
        }
        // else if (PyObject_IsInstance(value, THPVariableClass)) {
        //     PM_PyTensor_SetItem(chk_state, req->handle, old_value, value);
        // }
        else {
            // cuda::atomic_thread_fence(cuda::memory_order_seq_cst,
            //                 cuda::thread_scope_system);
            PM_PyTensor_SetItem(chk_state, req->handle, old_value, value);
            // printf("value is other types\n");
            // exit(1);
        }
        mp->ma_version_tag = 0;
        return 0;
    }
    else {
        printf("old value is null\n");
    }

 Fail:
    printf("set item failed\n");
    return -1;   
}

void PM_PyTensor_SetItem(ChkGlobalState& chk_state, int handle, PyObject *pm_value, PyObject *dram_value)
{
    chk_state.tensor_copy_num += 1;
    // std::cout << "start setItem for tensor object, tensor_copy_num is " << chk_state.tensor_copy_num  << std::endl;
    const auto& buffer = THPVariable_Unpack(dram_value);
    char* tptr = (char*)buffer.contiguous().data_ptr();
    const long long buffer_bytes = static_cast<long long int>(buffer.nbytes());

    PyListObject *pm_list = (PyListObject *)pm_value;
    Py_ssize_t len = Py_SIZE(pm_list);
    if (len != 2) {
        printf("tensor storage is in error\n");
        exit(1);
    }
    PyObject **objs = pm_list->ob_item;
    void *pm_ptr = PyLong_AsVoidPtr(objs[0]);
    long long pm_size = PyLong_AsLongLong(objs[1]);
    if (pm_size != buffer_bytes) {
        printf("tensor size is not equal\n");
        exit(1);
    }
    
    // printf("tensor addr: %p, size: %ld(%.2fKB), 16-base: 0x%x\n", pm_ptr, buffer_bytes, buffer_bytes / (1024.0), buffer_bytes);
    if (buffer.is_cuda()) {
        // printf("tensor is stored on GPU\n");
        auto& stream = get_extra_stream();
        // cudaMemcpy(pm_ptr, tptr, buffer_bytes, cudaMemcpyDefault);
        cudaMemcpyAsync(pm_ptr, tptr, buffer_bytes, cudaMemcpyDefault, stream);
        cudaStreamSynchronize(stream);
        if (chk_state.tensor_copy_num == chk_state.tensor_num) {
            // cudaStreamSynchronize(stream);
            chk_state.copy_flag = 1;
        }
        // chk_state.ready_events[handle]->RecordEvent();
    }
    else {
        // printf("tensor is stored on CPU\n");
        memcpy(pm_ptr, tptr, buffer_bytes);
        pmem_flush(pm_ptr, buffer_bytes);
    }
    chk_state.ready_save_flags[handle] = 1;
}

// copy PyObject *o in DRAM/GPU to PM, deep copy
PyObject *PM_PyDict_Write(PyObject *o)
{
    // printf("start dict copy\n");
    PyDictObject *mp;
    Py_ssize_t i, n;
    mp = (PyDictObject *)o;
    // std::cout << "write dict size is " << mp->ma_used << std::endl;
    if (mp->ma_used == 0) {
        /* The dict is empty; just return a new dict. */
        return PM_PyDict_New();
    }

    if (_PyDict_HasSplitTable(mp)) {

        printf("dict is splitted\n");

        PyDictObject *split_copy;
        Py_ssize_t size = shared_keys_usable_size(mp->ma_keys);
        PyDictKeysObject *newkeys;
        newkeys = pm_new_keys_write(mp->ma_keys);
        PyObject **newvalues;
        newvalues = pm_new_values(size);
        if (newvalues == NULL) {
            printf("allocate memory for newvalues failed\n");
            return NULL;
        }
        split_copy = PM_PyObject_GC_New(PyDictObject, &PyDict_Type);
        if (split_copy == NULL) {
            return NULL;
        }
        split_copy->ma_values = newvalues;
        split_copy->ma_keys = newkeys;
        split_copy->ma_used = mp->ma_used;
        split_copy->ma_version_tag = 0;
        for (i = 0, n = size; i < n; i++) {
            PyObject *value = mp->ma_values[i];
            if (PyObject_IsInstance(value, THPVariableClass))
            {
                split_copy->ma_values[i] = PM_PyTensor_Write(value);
            }
            else if (PyDict_Check(value)) {
                split_copy->ma_values[i] = PM_PyDict_Write(value);
            }
            else if (PyList_Check(value)) {
                split_copy->ma_values[i] = PM_PyList_Write(value);
            }
            else if (PyUnicode_Check(value)) {
                split_copy->ma_values[i] = PM_PyUnicode_Write(value);
            }
            else if (PyFloat_Check(value)) {
                double x = PyFloat_AS_DOUBLE(value);
                split_copy->ma_values[i] = PM_PyFloat_FromDouble_Write(x);
            }
            else if (PyTuple_Check(value)) {
                split_copy->ma_values[i] = PM_PyTuple_Write(value);
            }
            else if (PyLong_Check(value)) {
                split_copy->ma_values[i] = PM_PyLong_Write(value);
            }
            else {
                printf("value is other types\n");
                exit(1);
            }
        }
        return (PyObject *)split_copy;
    }
    
    if (mp->ma_values == NULL) {

        // printf("start clone combined dict\n");

        PyDictKeysObject *keys = pm_write_combined_dict_keys(mp);
        if (keys == NULL) {
            return NULL;
        }
        PyDictObject *new_dict = pm_new_dict(keys, NULL, 0, 0);
        if (new_dict == NULL) {
            /* In case of an error, `new_dict()` takes care of
               cleaning up `keys`. */
            return NULL;
        }

        new_dict->ma_used = mp->ma_used;

        return (PyObject *)new_dict;
    }
    printf("2* error\n");
    return NULL;
}

PyObject *PM_PyDict_New(void)
{
    return (PyObject *)pm_new_dict(Py_EMPTY_KEYS, empty_values, 0, 0);
}

Py_ssize_t All_PyDict_KeysSize(PyDictKeysObject *keys)
{
    return (sizeof(PyDictKeysObject)
            + DK_IXSIZE(keys) * DK_SIZE(keys)
            + DK_SIZE(keys) * sizeof(PyDictKeyEntry));
}

// int PM_PyDict_SetItem(PyObject *op, PyObject *key, PyObject *value)
// {
//     printf("start pydict set item\n");
//     PyDictObject *mp;
//     Py_hash_t hash;
//     mp = (PyDictObject *)op;
//     if (!PyUnicode_CheckExact(key) ||
//         (hash = ((PyASCIIObject *) key)->hash) == -1)
//     {
//         printf("1*\n");
//         hash = PyObject_Hash(key);
//         if (hash == -1)
//             return -1;
//     }
//     printf("2*\n");
//     return pm_insertdict(mp, key, hash, value);
// }

void PM_PyDict_Copy(ChkGlobalState& chk_state, PyObject *dst, PyObject *src)
{
    PyDictObject *mp;
    Py_ssize_t i, n;
    mp = (PyDictObject *)src;
    if (!PyDict_Check(dst)) {
        printf("dst is not dict type\n");
        exit(1);
    }
    if (_PyDict_HasSplitTable(mp)) {
        printf("dict is splitted\n");
        PyDictObject *split_copy = (PyDictObject *)dst;
        // pm_new_keys_copy(split_copy->ma_keys, mp->ma_keys);
        Py_ssize_t size = mp->ma_keys->dk_nentries;
        if (split_copy->ma_keys->dk_nentries != size) {
            printf("len of two dict is not equal\n");
            exit(1);
        }
        for (i = 0, n = size; i < n; i++) {
            PyObject *value = mp->ma_values[i];
            if (PyObject_IsInstance(value, THPVariableClass))
            {
                PM_PyTensor_Copy(chk_state, split_copy->ma_values[i], value);
            }
            else if (PyDict_Check(value)) {
                PM_PyDict_Copy(chk_state, split_copy->ma_values[i], value);
            }
            else if (PyList_Check(value)) {
                PM_PyList_Copy(chk_state, split_copy->ma_values[i], value);
            }
            else if (PyUnicode_Check(value)) {
                PM_PyUnicode_Copy(split_copy->ma_values[i], value);
            }
            else if (PyFloat_Check(value)) {
                double x = PyFloat_AS_DOUBLE(value);
                PM_PyFloat_FromDouble_Copy(split_copy->ma_values[i], x);
            }
            else if (PyLong_Check(value)) {
                PM_PyLong_Copy(split_copy->ma_values[i], value);
            }
            else if (PyTuple_Check(value)) {
                PM_PyTuple_Copy(chk_state, split_copy->ma_values[i], value);
            }
            else {
                printf("value is other types\n");
                exit(1);
            }
        }
    }

    if (mp->ma_values == NULL) {
        // printf("start copy combined dict\n");
        pm_copy_combined_dict_keys(chk_state, (PyDictObject *)dst, mp);
    }
}

PyObject *PM_PyList_Write(PyObject *a)
{
    // printf("start list copy\n");
    Py_ssize_t ilow = 0, ihigh = Py_SIZE((PyListObject *)a);
    PyListObject *np;
    PyObject **src, **dest;
    Py_ssize_t i, len;
    len = ihigh - ilow;
    if (len <= 0) {
        return PM_PyList_New(0);
    }
    np = (PyListObject *) PM_PyList_New(len);
    if (np == NULL)
        return NULL;

    PyListObject *al = (PyListObject *)a; 
    src = al->ob_item + ilow;
    dest = np->ob_item;
    for (i = 0; i < len; i++) {
        PyObject *value = src[i];
        if (PyObject_IsInstance(value, THPVariableClass))
        {
            dest[i] = PM_PyTensor_Write(value);
        }
        else if (PyDict_Check(value)) {
            dest[i] = PM_PyDict_Write(value);
        }
        else if (PyList_Check(value)) {
            dest[i] = PM_PyList_Write(value);
        }
        else if (PyUnicode_Check(value)) {
            dest[i] = PM_PyUnicode_Write(value);
        }
        else if (PyFloat_Check(value)) {
            double x = PyFloat_AS_DOUBLE(value);
            dest[i] = PM_PyFloat_FromDouble_Write(x);
        }
        else if (PyLong_Check(value)) {
            dest[i] = PM_PyLong_Write(value);
        }
        else if (PyTuple_Check(value)) {
            dest[i] = PM_PyTuple_Write(value);
        }
        else {
            printf("value is other types\n");
            exit(1);
        }
    }
    Py_SIZE(np) = len;
    return (PyObject *)np;
}

PyObject *PM_PyList_New(Py_ssize_t size)
{
    PyListObject *op;

    if (size < 0) {
        PyErr_BadInternalCall();
        return NULL;
    }

    op = PM_PyObject_GC_New(PyListObject, &PyList_Type);
    if (op == NULL) {
        return NULL;
    }
    if (size <= 0) {
        op->ob_item = NULL;
    }
    else {
        op->ob_item = (PyObject **) PM_PyObject_Calloc(size, sizeof(PyObject *));
        if (op->ob_item == NULL) {
            Py_DECREF(op);
            return NULL;
        }
    }
    Py_SIZE(op) = size;
    op->allocated = size;
    return (PyObject *) op;
}

void PM_PyList_Copy(ChkGlobalState& chk_state, PyObject *b, PyObject *a)
{
    if (!PyList_Check(b)) {
        printf("b is not list type\n");
        exit(1);
    }
    PyListObject *bl = (PyListObject *)b;
    PyListObject *al = (PyListObject *)a;
    Py_ssize_t ilow = 0, ihigh = Py_SIZE(al);
    PyObject **src, **dest;
    Py_ssize_t i, len;
    len = ihigh - ilow;
    src = al->ob_item + ilow;
    dest = bl->ob_item;
    if (Py_SIZE(al) != Py_SIZE(bl)) {
        printf("len of two list is not equal");
        exit(1);
    }
    for (i = 0; i < len; i++) {
        PyObject *value = src[i];
        
        if (PyDict_Check(value)) {
            PM_PyDict_Copy(chk_state, dest[i], value);
        }
        else if (PyList_Check(value)) {
            PM_PyList_Copy(chk_state, dest[i], value);
        }
        else if (PyUnicode_Check(value)) {
            PM_PyUnicode_Copy(dest[i], value);
        }
        else if (PyFloat_Check(value)) {
            double x = PyFloat_AS_DOUBLE(value);
            PM_PyFloat_FromDouble_Copy(dest[i], x);
        }
        else if (PyLong_Check(value)) {
            PM_PyLong_Copy(dest[i], value);
        }
        else if (PyTuple_Check(value)) {
            PM_PyTuple_Copy(chk_state, dest[i], value);
        }
        // else if (PyObject_IsInstance(value, THPVariableClass))
        // {
        //     PM_PyTensor_Copy(chk_state, dest[i], value);
        // }
        else {
            // printf("value is other types\n");
            // exit(1);
            PM_PyTensor_Copy(chk_state, dest[i], value);
        }
    }
}

PyObject *PM_PyTuple_Write(PyObject *aa)
{
    PyTupleObject *np;
    Py_ssize_t i;
    PyObject **src, **dest;
    PyTupleObject *a = (PyTupleObject *)aa;
    Py_ssize_t size = Py_SIZE(a);
    if (size == 0) {
        return PM_PyTuple_New(0);
    }

    np = (PyTupleObject *)PM_PyTuple_New(size);
    if (np == NULL) {
        return NULL;
    }
    src = a->ob_item;
    dest = np->ob_item;
    for (i = 0; i < size; i++) {
        PyObject *value = src[i];
        if (PyObject_IsInstance(value, THPVariableClass))
        {
            dest[i] = PM_PyTensor_Write(value);
        }
        else if (PyDict_Check(value)) {
            dest[i] = PM_PyDict_Write(value);
        }
        else if (PyList_Check(value)) {
            dest[i] = PM_PyList_Write(value);
        }
        else if (PyUnicode_Check(value)) {
            dest[i] = PM_PyUnicode_Write(value);
        }
        else if (PyFloat_Check(value)) {
            double x = PyFloat_AS_DOUBLE(value);
            dest[i] = PM_PyFloat_FromDouble_Write(x);
        }
        else if (PyLong_Check(value)) {
            dest[i] = PM_PyLong_Write(value);
        }
        else if (PyTuple_Check(value)) {
            dest[i] = PM_PyTuple_Write(value);
        }
        else {
            printf("value is other types\n");
            exit(1);
        }
    }
    return (PyObject *)np;
}

void PM_PyTuple_Read(PyObject *dram_list, PyObject *pm_list)
{
    PyTupleObject *bl = (PyTupleObject *)dram_list;
    PyTupleObject *al = (PyTupleObject *)pm_list;
    PyObject **src, **dest;
    Py_ssize_t i, size;
    size = Py_SIZE(al);
    src = al->ob_item;
    dest = bl->ob_item;
    if (Py_SIZE(al) != Py_SIZE(bl)) {
        printf("len of two tuple is not equal");
        exit(1);
    }
    for (i = 0; i < size; i++) {
        PyObject *value = dest[i];
        if (PyObject_IsInstance(value, THPVariableClass))
        {
            PM_PyTensor_Read(value, src[i]);
        }
        else if (PyDict_Check(value)) {
            PM_PyDict_Read(value, src[i]);
        }
        else if (PyList_Check(value)) {
            PM_PyList_Read(value, src[i]);
        }
        else if (PyUnicode_Check(value)) {
            PM_PyUnicode_Copy(value, src[i]);
        }
        else if (PyFloat_Check(value)) {
            double x = PyFloat_AS_DOUBLE(src[i]);
            PM_PyFloat_FromDouble_Copy(value, x);
        }
        else if (PyLong_Check(value)) {
            PM_PyLong_Copy(value, src[i]);
        }
        else if (PyTuple_Check(value)) {
            PM_PyTuple_Read(value, src[i]);
        }
        else {
            printf("value is other types\n");
            exit(1);
        }
    }
}

void PM_PyTuple_Copy(ChkGlobalState& chk_state, PyObject *b, PyObject *a)
{
    if (!PyTuple_Check(b)) {
        printf("b is not tuple type\n");
        exit(1);
    }
    // printf("start copy tuple type\n");
    PyTupleObject *bl = (PyTupleObject *)b;
    PyTupleObject *al = (PyTupleObject *)a;
    PyObject **src, **dest;
    Py_ssize_t i, size;
    size = Py_SIZE(al);
    src = al->ob_item;
    dest = bl->ob_item;
    if (Py_SIZE(al) != Py_SIZE(bl)) {
        printf("size of two tuple is not equal");
        exit(1);
    }
    for (i = 0; i < size; i++) {
        PyObject *value = src[i];
        
        if (PyDict_Check(value)) {
            PM_PyDict_Copy(chk_state, dest[i], value);
        }
        else if (PyList_Check(value)) {
            PM_PyList_Copy(chk_state, dest[i], value);
        }
        else if (PyUnicode_Check(value)) {
            PM_PyUnicode_Copy(dest[i], value);
        }
        else if (PyFloat_Check(value)) {
            double x = PyFloat_AS_DOUBLE(value);
            PM_PyFloat_FromDouble_Copy(dest[i], x);
        }
        else if (PyLong_Check(value)) {
            PM_PyLong_Copy(dest[i], value);
        }
        else if (PyTuple_Check(value)) {
            PM_PyTuple_Copy(chk_state, dest[i], value);
        }
        // else if (PyObject_IsInstance(value, THPVariableClass))
        // {
        //     PM_PyTensor_Copy(chk_state, dest[i], value);
        // }
        else {
            // printf("value is other types\n");
            // exit(1);
            PM_PyTensor_Copy(chk_state, dest[i], value);
        }
    }
}

PyObject *PM_PyTuple_New(Py_ssize_t size)
{
    PyTupleObject *op;
    if (size < 0) {
        PyErr_BadInternalCall();
        return NULL;
    }
    op = PM_PyObject_GC_NewVar(PyTupleObject, &PyTuple_Type, size);
    if (op == NULL)
        return NULL;
    Py_SIZE(op) = size;
    return (PyObject *)op;
}

PyObject *PM_PyFloat_FromDouble_Write(double fval)
{
    // printf("start create float object\n");
    PyFloatObject *op;
    op = (PyFloatObject *)PM_PyObject_Malloc(sizeof(PyFloatObject));
    if (!op) {
        return NULL;
    }
   /* Inline PyObject_New */
    (void)PyObject_INIT(op, &PyFloat_Type);
    op->ob_fval = fval;
    return (PyObject *) op;
}

void PM_PyFloat_FromDouble_Copy(PyObject *dst, double fval)
{
    if (!PyFloat_Check(dst)) {
        printf("dst is not float type\n");
        exit(1);
    }
    PyFloatObject *op = (PyFloatObject *)dst;
    op->ob_fval = fval;
}

PyObject *PM_PyLong_Write(PyObject *value)
{
    // printf("start create long object\n");
    PyLongObject *new_value;
    PyLongObject *src = (PyLongObject *)value;
    Py_ssize_t i = Py_SIZE(src);
    if (i < 0)  i = -(i);
    new_value = PM_PyLong_New(i);
    if (new_value != NULL) {
        Py_SIZE(new_value) = Py_SIZE(src);
        while (--i >= 0) {
            new_value->ob_digit[i] = src->ob_digit[i];
        }
    }
    return (PyObject *)new_value;
}

PyLongObject *PM_PyLong_New(Py_ssize_t size)
{
    PyLongObject *result;
    Py_ssize_t ndigits = size ? size : 1;
    result = (PyLongObject *)PM_PyObject_Malloc(offsetof(PyLongObject, ob_digit) +
                             ndigits*sizeof(digit));
    if (!result) {
        return NULL;
    }
    return (PyLongObject*)PyObject_INIT_VAR(result, &PyLong_Type, size);
}

void PM_PyLong_Copy(PyObject*dst, PyObject *value)
{
    if (!PyLong_Check(dst)) {
        printf("dst is not long type\n");
        exit(1);
    }
    PyLongObject *new_value = (PyLongObject *)dst;
    PyLongObject *src = (PyLongObject *)value;
    Py_ssize_t i = Py_SIZE(src);
    if (i < 0)  i = -(i);
    while (--i >= 0) {
        new_value->ob_digit[i] = src->ob_digit[i];
    }
}


#define MAX_UNICODE 0x10ffff
#define _PyUnicode_WSTR(op)                             \
    (((PyASCIIObject*)(op))->wstr)
#define _PyUnicode_WSTR_LENGTH(op)                      \
    (((PyCompactUnicodeObject*)(op))->wstr_length)
#define _PyUnicode_LENGTH(op)                           \
    (((PyASCIIObject *)(op))->length)
#define _PyUnicode_STATE(op)                            \
    (((PyASCIIObject *)(op))->state)
#define _PyUnicode_HASH(op)                             \
    (((PyASCIIObject *)(op))->hash)

PyObject *PM_PyUnicode_Write(PyObject *unicode)
{
    // printf("start create unicode object\n");
    Py_ssize_t length;
    PyObject *copy;

    length = PyUnicode_GET_LENGTH(unicode);
    copy = PM_PyUnicode_New(length, PyUnicode_MAX_CHAR_VALUE(unicode));
    if (!copy)
        return NULL;

    memcpy(PyUnicode_DATA(copy), PyUnicode_DATA(unicode),
              length * PyUnicode_KIND(unicode));
    return copy;
}

PyObject *PM_PyUnicode_New(Py_ssize_t size, Py_UCS4 maxchar)
{
    PyObject *obj;
    PyCompactUnicodeObject *unicode;
    void *data;
    enum PyUnicode_Kind kind;
    int is_sharing, is_ascii;
    Py_ssize_t char_size;
    Py_ssize_t struct_size;

    is_ascii = 0;
    is_sharing = 0;
    struct_size = sizeof(PyCompactUnicodeObject);
    if (maxchar < 128) {
        kind = PyUnicode_1BYTE_KIND;
        char_size = 1;
        is_ascii = 1;
        struct_size = sizeof(PyASCIIObject);
    }
    else if (maxchar < 256) {
        kind = PyUnicode_1BYTE_KIND;
        char_size = 1;
    }
    else if (maxchar < 65536) {
        kind = PyUnicode_2BYTE_KIND;
        char_size = 2;
        if (sizeof(wchar_t) == 2)
            is_sharing = 1;
    }
    else {
        if (maxchar > MAX_UNICODE) {
            PyErr_SetString(PyExc_SystemError,
                            "invalid maximum character passed to PyUnicode_New");
            return NULL;
        }
        kind = PyUnicode_4BYTE_KIND;
        char_size = 4;
        if (sizeof(wchar_t) == 4)
            is_sharing = 1;
    }

    /* Ensure we won't overflow the size. */
    if (size < 0) {
        PyErr_SetString(PyExc_SystemError,
                        "Negative size passed to PyUnicode_New");
        return NULL;
    }
    if (size > ((PY_SSIZE_T_MAX - struct_size) / char_size - 1))
        return NULL;

    obj = (PyObject *) PM_PyObject_Malloc(struct_size + (size + 1) * char_size);
    if (obj == NULL) {
        return NULL;
    }
    obj = PyObject_INIT(obj, &PyUnicode_Type);

    unicode = (PyCompactUnicodeObject *)obj;
    if (is_ascii)
        data = ((PyASCIIObject*)obj) + 1;
    else
        data = unicode + 1;
    _PyUnicode_LENGTH(unicode) = size;
    _PyUnicode_HASH(unicode) = -1;
    _PyUnicode_STATE(unicode).interned = 0;
    _PyUnicode_STATE(unicode).kind = kind;
    _PyUnicode_STATE(unicode).compact = 1;
    _PyUnicode_STATE(unicode).ready = 1;
    _PyUnicode_STATE(unicode).ascii = is_ascii;
    if (is_ascii) {
        ((char*)data)[size] = 0;
        _PyUnicode_WSTR(unicode) = NULL;
    }
    else if (kind == PyUnicode_1BYTE_KIND) {
        ((char*)data)[size] = 0;
        _PyUnicode_WSTR(unicode) = NULL;
        _PyUnicode_WSTR_LENGTH(unicode) = 0;
        unicode->utf8 = NULL;
        unicode->utf8_length = 0;
    }
    else {
        unicode->utf8 = NULL;
        unicode->utf8_length = 0;
        if (kind == PyUnicode_2BYTE_KIND)
            ((Py_UCS2*)data)[size] = 0;
        else /* kind == PyUnicode_4BYTE_KIND */
            ((Py_UCS4*)data)[size] = 0;
        if (is_sharing) {
            _PyUnicode_WSTR_LENGTH(unicode) = size;
            _PyUnicode_WSTR(unicode) = (wchar_t *)data;
        }
        else {
            _PyUnicode_WSTR_LENGTH(unicode) = 0;
            _PyUnicode_WSTR(unicode) = NULL;
        }
    }
    return obj;
}

void PM_PyUnicode_Copy(PyObject *dst, PyObject *unicode)
{
    if (!PyUnicode_Check(dst)) {
        printf("dst is not unicode type\n");
        exit(1);
    }
    Py_ssize_t length = PyUnicode_GET_LENGTH(unicode);

    memcpy(PyUnicode_DATA(dst), PyUnicode_DATA(unicode),
              length * PyUnicode_KIND(unicode));
}

void write_tensor_data(PyObject *pm_dict, PyObject *chk_dict, PyObject *buffer_name_list, PyObject *mapping) {
    PyDictObject *pm_model_dict, *pm_opt_state_dict;
    PyDictObject *g_model_dict, *g_opt_state_dict;
    pm_model_dict = (PyDictObject *)PyDict_GetItemString(pm_dict, "model");
    pm_opt_state_dict = (PyDictObject *)PyDict_GetItemString(PyDict_GetItemString(pm_dict, "optimizer"), "state");
    g_model_dict = (PyDictObject *)PyDict_GetItemString(chk_dict, "model");
    g_opt_state_dict = (PyDictObject *)PyDict_GetItemString(PyDict_GetItemString(chk_dict, "optimizer"), "state");
    Py_ssize_t n, i;
    Py_hash_t hash1, hash2;
    Py_ssize_t hashpos, ix;
    PyObject *param, *state_num;
    PyObject *pm_param_val, *pm_state_val, *g_param_val, *g_state_val;
    PyObject **value_addr;
    void *pm_param_ptr, *pm_state_ptr;
    long long pm_param_size, pm_state_size;
    char *g_param_ptr, *g_state_ptr;

    n = Py_SIZE((PyListObject *)buffer_name_list);
    PyObject **le = ((PyListObject *)buffer_name_list)->ob_item;
    for (i = 0; i < n; i++) {
        PyObject *buffer_name = le[i];
        if (!PyUnicode_CheckExact(buffer_name) || (hash1 = ((PyASCIIObject *)buffer_name)->hash) == -1) {
            hash1 = PyObject_Hash(buffer_name);
            if (hash1 == -1)
                exit(1);
        }
        ix = g_model_dict->ma_keys->dk_lookup(g_model_dict, buffer_name, hash1, &value_addr, &hashpos);
        if (ix == DKIX_ERROR)
            exit(1);
        g_param_val = *value_addr;            
        ix = pm_model_dict->ma_keys->dk_lookup(pm_model_dict, buffer_name, hash1, &value_addr, &hashpos);
        if (ix == DKIX_ERROR)
            exit(1);
        pm_param_val = *value_addr;
        if (!PyList_Check(pm_param_val)) {
            printf("param tensor store is not list, error\n");
            exit(1);
        }
        // pm_param_ptr = PyLong_AsVoidPtr((((PyListObject *)pm_param_val)->ob_item)[0]);
        pm_param_size = PyLong_AsLongLong((((PyListObject *)pm_param_val)->ob_item)[1]);
        const auto& g_param_buff = THPVariable_Unpack(g_param_val);
        g_param_ptr = (char *)g_param_buff.contiguous().data_ptr();
        if (pm_param_size != (static_cast<long long int>(g_param_buff.nbytes()))) {
            printf("tensor size is not equal\n");
            exit(1);
        }
        pm_param_ptr = PM_PyObject_Malloc_Gap(pm_param_size);
        if (g_param_buff.is_cuda()) {
            cudaMemcpy(pm_param_ptr, g_param_ptr, pm_param_size, cudaMemcpyDefault);
        }
        else {
            memcpy(pm_param_ptr, g_param_ptr, pm_param_size);
            pmem_flush(pm_param_ptr, pm_param_size);
        }
        (((PyListObject *)pm_param_val)->ob_item)[0] = PM_PyLong_FromVoidPtr(pm_param_ptr);
    }

    PyDictObject *mp = (PyDictObject *)mapping;
    PyDictKeyEntry *ep;
    if (_PyDict_HasSplitTable(mp)) {
        ep = DK_ENTRIES(mp->ma_keys);
        n = shared_keys_usable_size(mp->ma_keys);
        std::cout << "chk mapping size " << n << std::endl;
        for (i = 0; i < n; i++) {
            param = (&ep[i])->me_key;
            state_num = mp->ma_values[i];
            if (!PyUnicode_CheckExact(param) || (hash1 = ((PyASCIIObject *)param)->hash) == -1) {
                hash1 = PyObject_Hash(param);
                if (hash1 == -1)
                    exit(1);
            }
            if (!PyUnicode_CheckExact(state_num) || (hash2 = ((PyASCIIObject *)state_num)->hash) == -1) {
                hash2 = PyObject_Hash(state_num);
                if (hash2 == -1)
                    exit(1);
            }
            ix = g_model_dict->ma_keys->dk_lookup(g_model_dict, param, hash1, &value_addr, &hashpos);
            if (ix == DKIX_ERROR)
                exit(1);
            g_param_val = *value_addr;            
            ix = pm_model_dict->ma_keys->dk_lookup(pm_model_dict, param, hash1, &value_addr, &hashpos);
            if (ix == DKIX_ERROR)
                exit(1);
            pm_param_val = *value_addr;
            ix = g_opt_state_dict->ma_keys->dk_lookup(g_opt_state_dict, state_num, hash2, &value_addr, &hashpos);
            if (ix == DKIX_ERROR)
                exit(1);
            g_state_val = *value_addr;
            ix = pm_opt_state_dict->ma_keys->dk_lookup(pm_opt_state_dict, state_num, hash2, &value_addr, &hashpos);
            if (ix == DKIX_ERROR)
                exit(1);
            pm_state_val = *value_addr;
            if (!PyDict_Check(pm_state_val) || !PyDict_Check(g_state_val)) {
                printf("opt state for one num is not dict, error\n");
                exit(1);
            }
            Py_ssize_t n1 = ((PyDictObject *)pm_state_val)->ma_keys->dk_nentries;
            PyDictKeyEntry *p_ep = DK_ENTRIES(((PyDictObject *)pm_state_val)->ma_keys);
            PyDictKeyEntry *g_ep = DK_ENTRIES(((PyDictObject *)g_state_val)->ma_keys);
            for (Py_ssize_t j = n1 - 1; j >= 0; j--) {
                PyObject *p_oval = (&p_ep[j])->me_value;
                PyObject *g_oval = (&g_ep[j])->me_value;
                if (p_oval != NULL) {
                    if (!PyList_Check(p_oval)) {
                        continue;
                    }
                    pm_state_size = PyLong_AsLongLong((((PyListObject *)p_oval)->ob_item)[1]);
                    const auto& g_state_buff = THPVariable_Unpack(g_oval);
                    g_state_ptr = (char *)g_state_buff.contiguous().data_ptr();
                    if (pm_state_size != (static_cast<long long int>(g_state_buff.nbytes()))) {
                        printf("tensor size is not equal\n");
                        exit(1);
                    }
                    pm_state_ptr = PM_PyObject_Malloc_Gap(pm_state_size);
                    if (g_state_buff.is_cuda()) {
                        cudaMemcpy(pm_state_ptr, g_state_ptr, pm_state_size, cudaMemcpyDefault);
                    }
                    else {
                        memcpy(pm_state_ptr, g_state_ptr, pm_state_size);
                        pmem_flush(pm_state_ptr, pm_state_size);
                    }
                    (((PyListObject *)p_oval)->ob_item)[0] = PM_PyLong_FromVoidPtr(pm_state_ptr);
                }
            }

            // pm_param_ptr = PyLong_AsVoidPtr((((PyListObject *)pm_param_val)->ob_item)[0]);
            pm_param_size = PyLong_AsLongLong((((PyListObject *)pm_param_val)->ob_item)[1]);
            const auto& g_param_buff = THPVariable_Unpack(g_param_val);
            g_param_ptr = (char *)g_param_buff.contiguous().data_ptr();
            if (pm_param_size != (static_cast<long long int>(g_param_buff.nbytes()))) {
                printf("tensor size is not equal\n");
                exit(1);
            }
            pm_param_ptr = PM_PyObject_Malloc_Gap(pm_param_size);
            if (g_param_buff.is_cuda()) {
                cudaMemcpy(pm_param_ptr, g_param_ptr, pm_param_size, cudaMemcpyDefault);
            }
            else {
                memcpy(pm_param_ptr, g_param_ptr, pm_param_size);
                pmem_flush(pm_param_ptr, pm_param_size);
            }
            (((PyListObject *)pm_param_val)->ob_item)[0] = PM_PyLong_FromVoidPtr(pm_param_ptr);
        }
    }
    else {
        ep = DK_ENTRIES(mp->ma_keys);
        n = mp->ma_keys->dk_nentries;
        std::cout << "chk mapping size " << n << std::endl;
        for (i = n - 1; i >= 0; i--) {
            param = (&ep[i])->me_key;
            state_num = (&ep[i])->me_value;
            if (!PyUnicode_CheckExact(param) || (hash1 = ((PyASCIIObject *)param)->hash) == -1) {
                hash1 = PyObject_Hash(param);
                if (hash1 == -1)
                    exit(1);
            }
            if (!PyUnicode_CheckExact(state_num) || (hash2 = ((PyASCIIObject *)state_num)->hash) == -1) {
                hash2 = PyObject_Hash(state_num);
                if (hash2 == -1)
                    exit(1);
            }
            if (PyLong_Check(state_num)) {
                std::cout << "state num " << PyLong_AsLong(state_num) << std::endl; 
            }
            ix = g_model_dict->ma_keys->dk_lookup(g_model_dict, param, hash1, &value_addr, &hashpos);
            if (ix == DKIX_ERROR)
                exit(1);
            g_param_val = *value_addr;            
            ix = pm_model_dict->ma_keys->dk_lookup(pm_model_dict, param, hash1, &value_addr, &hashpos);
            if (ix == DKIX_ERROR)
                exit(1);
            pm_param_val = *value_addr;
            ix = g_opt_state_dict->ma_keys->dk_lookup(g_opt_state_dict, state_num, hash2, &value_addr, &hashpos);
            if (ix == DKIX_ERROR)
                exit(1);
            g_state_val = *value_addr;
            ix = pm_opt_state_dict->ma_keys->dk_lookup(pm_opt_state_dict, state_num, hash2, &value_addr, &hashpos);
            if (ix == DKIX_ERROR)
                exit(1);
            pm_state_val = *value_addr;
            if (!PyList_Check(pm_param_val)) {
                printf("param tensor store is not list, error\n");
                exit(1);
            }
            if (!PyDict_Check(pm_state_val) || !PyDict_Check(g_state_val)) {
                printf("opt state for one num is not dict, error\n");
                exit(1);
            }
            Py_ssize_t n1 = ((PyDictObject *)pm_state_val)->ma_keys->dk_nentries;
            PyDictKeyEntry *p_ep = DK_ENTRIES(((PyDictObject *)pm_state_val)->ma_keys);
            PyDictKeyEntry *g_ep = DK_ENTRIES(((PyDictObject *)g_state_val)->ma_keys);
            for (Py_ssize_t j = 0; j < n1; j++) {
                PyObject *p_oval = (&p_ep[j])->me_value;
                PyObject *g_oval = (&g_ep[j])->me_value;
                if (p_oval != NULL) {
                    if (!PyList_Check(p_oval)) {
                        // printf("opt state tensor store is not list, error\n");
                        // exit(1);
                        continue;
                    }
                    pm_state_size = PyLong_AsLongLong((((PyListObject *)p_oval)->ob_item)[1]);
                    const auto& g_state_buff = THPVariable_Unpack(g_oval);
                    g_state_ptr = (char *)g_state_buff.contiguous().data_ptr();
                    if (pm_state_size != (static_cast<long long int>(g_state_buff.nbytes()))) {
                        printf("tensor size is not equal\n");
                        exit(1);
                    }
                    pm_state_ptr = PM_PyObject_Malloc_Gap(pm_state_size);
                    if (g_state_buff.is_cuda()) {
                        cudaMemcpy(pm_state_ptr, g_state_ptr, pm_state_size, cudaMemcpyDefault);
                    }
                    else {
                        memcpy(pm_state_ptr, g_state_ptr, pm_state_size);
                        pmem_flush(pm_state_ptr, pm_state_size);
                    }
                    (((PyListObject *)p_oval)->ob_item)[0] = PM_PyLong_FromVoidPtr(pm_state_ptr);
                }
            }

            // pm_param_ptr = PyLong_AsVoidPtr((((PyListObject *)pm_param_val)->ob_item)[0]);
            pm_param_size = PyLong_AsLongLong((((PyListObject *)pm_param_val)->ob_item)[1]);
            const auto& g_param_buff = THPVariable_Unpack(g_param_val);
            g_param_ptr = (char *)g_param_buff.contiguous().data_ptr();
            if (pm_param_size != (static_cast<long long int>(g_param_buff.nbytes()))) {
                printf("tensor size is not equal\n");
                exit(1);
            }
            pm_param_ptr = PM_PyObject_Malloc_Gap(pm_param_size);
            if (g_param_buff.is_cuda()) {
                cudaMemcpy(pm_param_ptr, g_param_ptr, pm_param_size, cudaMemcpyDefault);
            }
            else {
                memcpy(pm_param_ptr, g_param_ptr, pm_param_size);
                pmem_flush(pm_param_ptr, pm_param_size);
            }
            (((PyListObject *)pm_param_val)->ob_item)[0] = PM_PyLong_FromVoidPtr(pm_param_ptr);
        }
    }   
}

PyObject *PM_PyTensor_Write(PyObject *value)
{
    // const at::Tensor&
    const auto& buffer = THPVariable_Unpack(value);
    // std::cout << "write tensor is " << buffer << std::endl;
    // if (buffer.is_contiguous())
    //     std::cout << "buffer is contiguous" << std::endl;
    // else
    //     std::cout << "buffer is not contiguous" << std::endl;
    
    char* tptr = (char*)buffer.contiguous().data_ptr();
    const long long buffer_bytes = static_cast<long long int>(buffer.nbytes());
    // const long long buffer_bytes1 = static_cast<long long int>(buffer.numel() * buffer.element_size());
    // std::cout << "buffer_bytes: " << buffer_bytes << ", buffer_bytes1: " << buffer_bytes1 << std::endl;
    void *tdst = NULL;
    if (chk_way != 2) {
        tdst = PM_PyObject_Malloc_Gap(buffer_bytes);

        if (buffer.is_cuda()) {
            cudaMemcpy(tdst, tptr, buffer_bytes, cudaMemcpyDefault);
        }
        else {
            memcpy(tdst, tptr, buffer_bytes);
            pmem_flush(tdst, buffer_bytes);
        }
    }
    Py_ssize_t len = 2;
    PyObject **list_item;
    PyListObject *np = (PyListObject *) PM_PyList_New(len);
    list_item = np->ob_item;
    if (chk_way != 2)
        list_item[0] = PM_PyLong_FromVoidPtr(tdst);
    list_item[1] = PM_PyLong_FromLongLong(buffer_bytes);
    return (PyObject *)np;
}

void PM_PyTensor_Copy(ChkGlobalState& chk_state, PyObject *pm_value, PyObject *dram_value)
{
    chk_state.tensor_copy_num += 1;
    // std::cout << "start setItem for tensor object, tensor_copy_num is " << chk_state.tensor_copy_num  << std::endl;
    const auto& buffer = THPVariable_Unpack(dram_value);
    char* tptr = (char*)buffer.contiguous().data_ptr();
    const long long buffer_bytes = static_cast<long long int>(buffer.nbytes());

    PyListObject *pm_list = (PyListObject *)pm_value;
    Py_ssize_t len = Py_SIZE(pm_list);
    if (len != 2) {
        printf("tensor storage is in error\n");
        exit(1);
    }
    PyObject **objs = pm_list->ob_item;
    void *pm_ptr = PyLong_AsVoidPtr(objs[0]);
    long long pm_size = PyLong_AsLongLong(objs[1]);
    if (pm_size != buffer_bytes) {
        printf("tensor size is not equal\n");
        exit(1);
    }
    // printf("tensor addr: %p, size: %ld(%.2fKB), 16-base: 0x%x\n", pm_ptr, buffer_bytes, buffer_bytes / (1024.0), buffer_bytes);
    if (buffer.is_cuda()) {
        // printf("tensor is stored on GPU\n");
        // auto stream = c10::cuda::getCurrentCUDAStream();
        auto& stream = get_extra_stream();
        // cudaMemcpy(pm_ptr, tptr, buffer_bytes, cudaMemcpyDefault);
        cudaMemcpyAsync(pm_ptr, tptr, buffer_bytes, cudaMemcpyDefault, stream);
        cudaStreamSynchronize(stream);
        if (chk_state.tensor_copy_num == chk_state.tensor_num) {
            // cudaStreamSynchronize(stream);
            chk_state.copy_flag = 1;
        }
        // auto ready_event = RecordReadyEvent();
        // chk_state.handle ++;
        // {
        //     std::lock_guard<std::mutex> guard(chk_state.event_mutex);
        //     chk_state.ready_events[chk_state.handle] = ready_event;
        // }
    }
    else {
        // printf("tensor is stored on CPU\n");
        memcpy(pm_ptr, tptr, buffer_bytes);
        pmem_flush(pm_ptr, buffer_bytes);
    }
}

bool poll_all_events(ChkGlobalState& chk_state)
{
    std::lock_guard<std::mutex> guard(chk_state.event_mutex);
    std::cout << "poll " << chk_state.ready_events.size() << " events" << std::endl;
    auto iter = chk_state.ready_events.begin();
    while (iter != chk_state.ready_events.end()) {
        auto ready_event = iter->second;
        int ready = ready_event->Ready() ? 1 : 0;
        if (ready) {
            chk_state.ready_events.erase(iter ++);
        }
        else {
            std::cout << "poll_all_events, not all events are ready" << std::endl;
            return false;
        }
    }
    std::cout << "all events are ready, current events size is " << chk_state.ready_events.size() << std::endl;
    return true;
}

bool poll_handle(ChkGlobalState& chk_state, int handle)
{
    std::lock_guard<std::mutex> guard(chk_state.event_mutex);
    if (chk_state.ready_save_flags.find(handle) == chk_state.ready_save_flags.end()) {
        std::cout << "handle " << handle << " does not exist" << std::endl;
        exit(1);
    }
    if (chk_state.ready_save_flags[handle] == 0)
        return false;
    // if (chk_state.ready_events.find(handle) == chk_state.ready_events.end()) {
    //     std::cout << "handle does not exist" << std::endl;
    //     exit(1);
    // }
    // auto ready_event = chk_state.ready_events[handle];
    // int ready = ready_event->Ready() ? 1 : 0;
    // if (ready) {
    //     chk_state.ready_events.erase(handle);
    // }
    // else {
    //     // std::cout << "poll handle " << handle << ", not all events are ready" << std::endl;
    //     return false;
    // }
    chk_state.ready_save_flags.erase(handle);
    return true;
}

PyObject *PM_PyLong_FromVoidPtr(void *ptr)
{
    unsigned long ival = (unsigned long)(uintptr_t)ptr;
    PyLongObject *v;
    unsigned long t;
    int ndigits = 0;
    /* Count the number of Python digits. */
    t = (unsigned long)ival;
    while (t) {
        ++ndigits;
        t >>= PyLong_SHIFT;
    }
    v = PM_PyLong_New(ndigits);
    if (v != NULL) {
        digit *p = v->ob_digit;
        Py_SIZE(v) = ndigits;
        while (ival) {
            *p++ = (digit)(ival & PyLong_MASK);
            ival >>= PyLong_SHIFT;
        }
    }
    return (PyObject *)v;
}

PyObject *PM_PyLong_FromLongLong(long long ival)
{
    PyLongObject *v;
    unsigned long long abs_ival;
    unsigned long long t;  /* unsigned so >> doesn't propagate sign bit */
    int ndigits = 0;
    if (ival < 0) {
        printf("tensor bytes cannot be negative!\n");
        exit(1);
    }
    else {
        abs_ival = (unsigned long long)ival;
    }
    t = abs_ival;
    while (t) {
        ++ndigits;
        t >>= PyLong_SHIFT;
    }
    v = PM_PyLong_New(ndigits);
    if (v != NULL) {
        digit *p = v->ob_digit;
        Py_SIZE(v) = ndigits;
        t = abs_ival;
        while (t) {
            *p++ = (digit)(t & PyLong_MASK);
            t >>= PyLong_SHIFT;
        }
    }
    return (PyObject *)v;
}

static void pm_read_combined_dict_keys(PyDictObject *dram_mp, PyDictObject *pm_mp)
{
    PyDictKeyEntry *ep0 = DK_ENTRIES(dram_mp->ma_keys);
    // PyDictKeyEntry *pm_ep0 = DK_ENTRIES(pm_mp->ma_keys);
    Py_ssize_t n = pm_mp->ma_keys->dk_nentries;
    Py_ssize_t ix;
    Py_hash_t hash;
    PyObject *pm_value;
    PyObject **value_addr;
    if (dram_mp->ma_keys->dk_nentries != n) {
        printf("size of two dict is not equal");
        exit(1);
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyDictKeyEntry *entry = &ep0[i];
        PyObject *key = entry->me_key;
        if (!PyUnicode_CheckExact(key) ||
        (hash = ((PyASCIIObject *) key)->hash) == -1)
        {
            hash = PyObject_Hash(key);
            if (hash == -1)
                exit(1);
        }
        ix = pm_mp->ma_keys->dk_lookup(pm_mp, key, hash, &value_addr, NULL);
        pm_value = *value_addr;
        if (pm_value != NULL) {
            if (PyObject_IsInstance(entry->me_value, THPVariableClass))
            {
                PM_PyTensor_Read(entry->me_value, pm_value);
            }
            else if (PyDict_Check(entry->me_value)) {
                PM_PyDict_Read(entry->me_value, pm_value);
            }
            else if (PyList_Check(entry->me_value)) {
                PM_PyList_Read(entry->me_value, pm_value);
            }
            else if (PyUnicode_Check(entry->me_value)) {
                PM_PyUnicode_Copy(entry->me_value, pm_value);
            }
            else if (PyFloat_Check(entry->me_value)) {
                double x = PyFloat_AS_DOUBLE(pm_value);
                PM_PyFloat_FromDouble_Copy(entry->me_value, x);
            }
            else if (PyLong_Check(entry->me_value)) {
                PM_PyLong_Copy(entry->me_value, pm_value);
            }
            else if (PyTuple_Check(entry->me_value)) {
                PM_PyTuple_Read(entry->me_value, pm_value);
            }
            else {
                printf("value is other types\n");
                exit(1);
            }
            // entry->me_hash = pm_entry->me_hash;
        }
    }
}

void PM_PyDict_Read(PyObject *dram_dict, PyObject *pm_dict)
{
    PyDictObject *dram_mp, *pm_mp;
    Py_ssize_t i, n;
    dram_mp = (PyDictObject *)dram_dict;
    pm_mp = (PyDictObject *)pm_dict;
    if (dram_mp->ma_used != pm_mp->ma_used) {
        printf("ma_used of two dict is not equal\n");
        exit(1);
    }
    if (_PyDict_HasSplitTable(dram_mp)) {
        if (pm_mp->ma_values == NULL) {
            printf("one of two dict is splitted and the other is not\n");
            exit(1);
        }
        Py_ssize_t size = pm_mp->ma_keys->dk_nentries;
        if (dram_mp->ma_keys->dk_nentries != size) {
            printf("len of two dict is not equal\n");
            exit(1);
        }

        for (i = 0, n = size; i < n; i++) {
            PyObject *value = dram_mp->ma_values[i];
            if (PyObject_IsInstance(value, THPVariableClass))
            {
                PM_PyTensor_Read(value, pm_mp->ma_values[i]);
            }
            else if (PyDict_Check(value)) {
                PM_PyDict_Read(value, pm_mp->ma_values[i]);
            }
            else if (PyList_Check(value)) {
                PM_PyList_Read(value, pm_mp->ma_values[i]);
            }
            else if (PyUnicode_Check(value)) {
                PM_PyUnicode_Copy(value, pm_mp->ma_values[i]);
            }
            else if (PyFloat_Check(value)) {
                double x = PyFloat_AS_DOUBLE(pm_mp->ma_values[i]);
                PM_PyFloat_FromDouble_Copy(value, x);
            }
            else if (PyLong_Check(value)) {
                PM_PyLong_Copy(value, pm_mp->ma_values[i]);
            }
            else if (PyTuple_Check(value)) {
                PM_PyTuple_Read(value, pm_mp->ma_values[i]);
            }
            else {
                printf("value is other types\n");
                exit(1);
            }
        }
    }
    if (dram_mp->ma_values == NULL) {
        if (pm_mp->ma_values != NULL) {
            printf("one of two dict is splitted and the other is not\n");
            exit(1);
        }
        pm_read_combined_dict_keys(dram_mp, pm_mp);
    }
}

void PM_PyList_Read(PyObject *dram_list, PyObject *pm_list)
{
    PyListObject *bl = (PyListObject *)dram_list;
    PyListObject *al = (PyListObject *)pm_list;
    Py_ssize_t ilow = 0, ihigh = Py_SIZE(al);
    PyObject **src, **dest;
    Py_ssize_t i, len;
    len = ihigh - ilow;
    src = al->ob_item + ilow;
    dest = bl->ob_item;
    if (Py_SIZE(al) != Py_SIZE(bl)) {
        printf("len of two list is not equal");
        exit(1);
    }
    for (i = 0; i < len; i++) {
        PyObject *value = dest[i];
        if (PyObject_IsInstance(value, THPVariableClass))
        {
            PM_PyTensor_Read(value, src[i]);
        }
        else if (PyDict_Check(value)) {
            PM_PyDict_Read(value, src[i]);
        }
        else if (PyList_Check(value)) {
            PM_PyList_Read(value, src[i]);
        }
        else if (PyUnicode_Check(value)) {
            PM_PyUnicode_Copy(value, src[i]);
        }
        else if (PyFloat_Check(value)) {
            double x = PyFloat_AS_DOUBLE(src[i]);
            PM_PyFloat_FromDouble_Copy(value, x);
        }
        else if (PyLong_Check(value)) {
            PM_PyLong_Copy(value, src[i]);
        }
        else if (PyTuple_Check(value)) {
            PM_PyTuple_Read(value, src[i]);
        }
        else {
            printf("value is other types\n");
            exit(1);
        }
    }
}

void PM_PyTensor_Read(PyObject *dram_value, PyObject *pm_value)
{
    const auto& buffer = THPVariable_Unpack(dram_value);
    char* tptr = (char*)buffer.contiguous().data_ptr();
    const long long buffer_bytes = static_cast<long long int>(buffer.nbytes());

    PyListObject *pm_list = (PyListObject *)pm_value;
    Py_ssize_t len = Py_SIZE(pm_list);
    if (len != 2) {
        printf("tensor storage is in error\n");
        exit(1);
    }
    PyObject **objs = pm_list->ob_item;
    void *pm_ptr = PyLong_AsVoidPtr(objs[0]);
    long long pm_size = PyLong_AsLongLong(objs[1]);
    if (pm_size != buffer_bytes) {
        printf("tensor size is not equal\n");
        exit(1);
    }
    // std::cout << "buffer_bytes: " << buffer_bytes << std::endl;
    if (buffer.is_cuda()) {
        // printf("tensor is stored on GPU\n");
        // cudaMemcpy(tptr, pm_ptr, buffer_bytes, cudaMemcpyDefault);
        auto stream = c10::cuda::getCurrentCUDAStream();
        cudaMemcpyAsync(tptr, pm_ptr, buffer_bytes, cudaMemcpyDefault, stream);
        cudaStreamSynchronize(stream);
    }
    else {
        // printf("tensor is stored on CPU\n");
        // copy tensor data to DRAM
        memcpy(tptr, pm_ptr, buffer_bytes);
    }
    // std::cout << "read tensor is " << buffer << std::endl;
}
