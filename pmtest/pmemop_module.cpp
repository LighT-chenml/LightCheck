#include <Python.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/mman.h>
#include <errno.h>
#include <sys/mman.h>
#include <libpmem.h>
#include <netdb.h>
#include <thread>
#include <mutex>
#include <iostream>
#include <memory>
#include <vector>
#include "pmemop_object.h"
#include "pmemop_alloc.h"
#include "ready_event.h"

// PyObject *glo_dict = NULL;
// std::vector<PyObject *> glo_vec;

ChkGlobalState chk_global;

static PyObject* synchronize(PyObject *self, PyObject *args) {
    while (true) {
        if (chk_global.save_flag == 1) break;
        std::this_thread::yield();
    }

    while (true) {
        if (poll_all_events(chk_global) == true) break;
        std::this_thread::yield();
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* mmap_pmem(PyObject *self, PyObject *args) {
    PM_Map();
    create_extra_stream();
    
    chk_global.initial_done = false;
    chk_global.tensor_num = 0;
    chk_global.chk_thread = std::thread(ChkThreadLoop, std::ref(chk_global));

    while (!chk_global.initial_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    chk_global.chk_thread.detach();
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* init_model_state(PyObject *self, PyObject *args) {
    printf("go into init_model_state\n");
    PyObject *chk_dict = NULL, *buffer_list = NULL, *mapping = NULL, *chk_way = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyDict_Type, &chk_dict, &PyList_Type, &buffer_list, &PyDict_Type, &mapping, &PyLong_Type, &chk_way))
        return NULL;
    long way = PyLong_AsLong(chk_way);
    set_chk_way(way);
    for (int i = 0; i < 2; i++) {
        chk_global.pm_dict[i] = PM_PyDict_Write(chk_dict);
        if (way == 2)
            write_tensor_data(chk_global.pm_dict[i], chk_dict, buffer_list, mapping);
        set_dram_alloc();
        chk_global.dram_dict[i] = PM_PyDict_Write(chk_global.pm_dict[i]);
        unset_dram_alloc();
        chk_global.model_dict[i] = PyDict_GetItemString(chk_global.pm_dict[i], "model");
        chk_global.opt_dict[i] = PyDict_GetItemString(chk_global.pm_dict[i], "optimizer");
        chk_global.opt_state_dict[i] = PyDict_GetItemString(chk_global.opt_dict[i], "state");
        change_alloc_flag();
    }
    chk_global.save_flag = 1;
    chk_global.chk_flag = 0;
    chk_global.handle = 0;
    chk_global.tensor_copy_num = 0;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* unmmap_pmem(PyObject *self, PyObject *args) {
    // ChkRequest req;
    // req.req_type = 5;
    // req.dict = NULL;
    auto req_op = std::make_shared<ChkRequest>(5, nullptr, nullptr, nullptr, 0);

    // chk_global.chk_que.push(std::move(req));
    chk_global.chk_que.push(req_op);
    chk_global.que_cond.notify_one();
    synchronize(NULL, NULL);
    sleep(3);
    // chk_global.chk_thread.join();
    PM_Unmap();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* save_dict(PyObject *self, PyObject *args) {    
    // std::cout << "main thread id=" << std::this_thread::get_id() << std::endl;
    PyObject *dict = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict))
        return NULL;
    {
        std::lock_guard<std::mutex> guard(chk_global.save_mutex);
        if (chk_global.save_flag == 0) {
            std::cout << "last save_dict operation is not finished" << std::endl;
            return NULL;
        }
        // std::cout << "main thread set save_flag 0" << std::endl;
        chk_global.save_flag = 0;
    }
    chk_global.chk_flag = (chk_global.chk_flag + 1) % 2;
    chk_global.tensor_copy_num = 0;
    chk_global.handle = 0;
    // std::cout << "main thread, dict size is " << ((PyDictObject *)dict)->ma_used << std::endl;
    // ChkRequest req;
    // req.req_type = 0;
    // Py_INCREF(dict);
    // req.dict = std::move(dict);
    // req.dict = dict;
    auto req_op = std::make_shared<ChkRequest>(0, std::move(dict), nullptr, nullptr, 0);

    std::lock_guard<std::mutex> lk(chk_global.que_mutex);
    // chk_global.chk_que.push(std::move(req));
    chk_global.chk_que.push(req_op);
    chk_global.que_cond.notify_one();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* query_save(PyObject *self, PyObject *args) {
    // printf("query save\n");
    int flag;
    {
        std::lock_guard<std::mutex> guard(chk_global.save_mutex);
        flag = chk_global.save_flag;
    }
    if (flag == 0) {
        Py_RETURN_FALSE;
    }
    if (poll_all_events(chk_global) == false)
        Py_RETURN_FALSE;
    printf("query save complete\n");
    Py_RETURN_TRUE;
}

static PyObject* query_handle(PyObject *self, PyObject *args) {
    int handle;

    // if (chk_global.tensor_num != 0 && chk_global.copy_flag == 0) Py_RETURN_FALSE;

    if (!PyArg_ParseTuple(args, "i", &handle))
        return NULL;
    if (poll_handle(chk_global, handle) == false)
        Py_RETURN_FALSE;
    // printf("poll handle %d finish\n", handle);
    Py_RETURN_TRUE;
}

static PyObject* wait_handle(PyObject *self, PyObject *args) {
    int handle;
    if (!PyArg_ParseTuple(args, "i", &handle))
        return NULL;
    // printf("wait handle %d finish\n", handle);

    // while (true) {
    //     if (chk_global.copy_flag == 1) break;
    //     std::this_thread::yield();
    // }

    while (true) {
        if (poll_handle(chk_global, handle)) break;
        std::this_thread::yield();
    }
    // printf("end, handle %d finish\n", handle);
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* load_dict(PyObject *self, PyObject *args) {
    printf("go into load dict function\n");
    PyObject *dict = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict))
        return NULL;
    
    PM_PyDict_Read(dict, chk_global.pm_dict[0]);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* use_new_dict(PyObject *self, PyObject *args) {
    chk_global.chk_flag = (chk_global.chk_flag + 1) % 2;
    if (chk_global.tensor_copy_num != 0)
        chk_global.tensor_num = chk_global.tensor_copy_num;
    printf("tensor num:%d, tensor copy num:%d\n", chk_global.tensor_num, chk_global.tensor_copy_num);
    chk_global.tensor_copy_num = 0;
    chk_global.copy_flag = 0;
    // chk_global.handle = 0;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* set_additional_item(PyObject *self, PyObject *args) {
    PyObject *key = NULL;
    PyObject *value = NULL;
    if (!PyArg_ParseTuple(args, "OO", &key, &value))
        return NULL;
    
    // ChkRequest req;
    // req.req_type = 1;
    // req.key = std::move(key);
    // req.value = std::move(value);

    auto req_op = std::make_shared<ChkRequest>(1, nullptr, std::move(key), std::move(value), 0);

    std::lock_guard<std::mutex> lk(chk_global.que_mutex);
    // chk_global.chk_que.push(std::move(req));
    chk_global.chk_que.push(req_op);
    chk_global.que_cond.notify_one();

    // return PyLong_FromLong(0);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* set_model_item(PyObject *self, PyObject *args) {
    PyObject *key = NULL;
    PyObject *value = NULL;
    if (!PyArg_ParseTuple(args, "OO", &key, &value))
        return NULL;
    // if (!PyObject_IsInstance(value, THPVariableClass)) {
    //     printf("model item is not tensor\n");
    //     exit(1);
    // }
    
    // ChkRequest req;
    int handle = 0;
    // req.req_type = 2;
    
    // auto ready_event = RecordReadyEvent();
    {
        std::lock_guard<std::mutex> guard(chk_global.event_mutex);
        chk_global.handle ++;
        handle = chk_global.handle;
        // chk_global.ready_events[handle] = ready_event;
    }
    
    chk_global.ready_save_flags[handle] = 0;
    // req.handle = handle;
    // req.key = std::move(key);
    // req.value = std::move(value);

    auto req_op = std::make_shared<ChkRequest>(2, nullptr, std::move(key), std::move(value), handle);
    
    std::lock_guard<std::mutex> lk(chk_global.que_mutex);
    // chk_global.chk_que.push(std::move(req));
    chk_global.chk_que.push(req_op);
    // if (!chk_global.chk_que.empty())
    chk_global.que_cond.notify_one();

    return PyLong_FromLong(handle);
}

static PyObject* set_opt_param_groups(PyObject *self, PyObject *args) {
    PyObject *param_groups = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &param_groups))
        return NULL;

    // ChkRequest req;
    // req.req_type = 3;
    PyObject* key = PyUnicode_FromString("param_groups");
    // req.key = std::move(key);
    // req.value = std::move(param_groups);

    auto req_op = std::make_shared<ChkRequest>(3, nullptr, std::move(key), std::move(param_groups), 0);

    std::lock_guard<std::mutex> lk(chk_global.que_mutex);
    // chk_global.chk_que.push(std::move(req));
    chk_global.chk_que.push(req_op);
    chk_global.que_cond.notify_one();

    // return PyLong_FromLong(0);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* set_opt_state(PyObject *self, PyObject *args) {
    PyObject *key = NULL;
    PyObject *value = NULL;
    if (!PyArg_ParseTuple(args, "OO", &key, &value))
        return NULL;
    // ChkRequest req;
    // req.req_type = 4;
    // req.key = std::move(key);
    // req.value = std::move(value);

    auto req_op = std::make_shared<ChkRequest>(4, nullptr, std::move(key), std::move(value), 0);

    std::lock_guard<std::mutex> lk(chk_global.que_mutex);
    // chk_global.chk_que.push(std::move(req));
    chk_global.chk_que.push(req_op);
    chk_global.que_cond.notify_one();

    // return PyLong_FromLong(0);
    Py_INCREF(Py_None);
    return Py_None;
}

// static PyObject* set_item(PyObject *self, PyObject *args) {
//     printf("start set_item\n");
//     PyObject *key = NULL;
//     PyObject *value = NULL;
//     if (!PyArg_ParseTuple(args, "OO", &key, &value))
//         return NULL;
//     PyObject *dict = chk_global.pm_dict[0];
//     if (!PyDict_Check(dict)) {
//         printf("this is not a dict\n");
//         exit(1);
//     }
//     printf("refcnt: key is %ld, value is %ld\n", key->ob_refcnt, value->ob_refcnt);
//     if (PM_PyDict_SetItem(dict, key, value)) {
//         printf("set item failed\n");
//         exit(1);
//     }
//     printf("refcnt: key is %ld, value is %ld\n", key->ob_refcnt, value->ob_refcnt);
//     Py_INCREF(Py_None);
//     return Py_None;
// }

static PyMethodDef methods[] = {
  { "mmap_pmem", mmap_pmem, METH_NOARGS, "mmap pmem" },
  { "unmmap_pmem", unmmap_pmem, METH_NOARGS, "unmmap pmem"},
  { "save_dict", save_dict, METH_VARARGS, "write pmem" },
  { "load_dict", load_dict, METH_VARARGS, "read pmem"},
  // { "set_item", set_item, METH_VARARGS, "set dict item"},
  { "synchronize", synchronize, METH_NOARGS, "synchronize write pmem"},
  { "query_save", query_save, METH_NOARGS, "query save"},
  { "query_handle", query_handle, METH_VARARGS, "query handle"},
  { "wait_handle", wait_handle, METH_VARARGS, "wait handle"},
  { "use_new_dict", use_new_dict, METH_NOARGS, "use_new_dict"},
  { "set_additional_item", set_additional_item, METH_VARARGS, "set_additional_item"},
  { "set_model_item", set_model_item, METH_VARARGS, "set_model_item"},
  { "init_model_state", init_model_state, METH_VARARGS, "init_model_state"},
  { "set_opt_param_groups", set_opt_param_groups, METH_VARARGS, "set_opt_param_groups"},
  { "set_opt_state", set_opt_state, METH_VARARGS, "set_opt_state"},
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef pmemopmodule = {
  PyModuleDef_HEAD_INIT,    // always required
  "pmemop",                 // module name
  "pmem operation module",  // description
  -1,                       // module size (-1 indicates we don't use this feature)
  methods,                  // method table
};

PyMODINIT_FUNC PyInit_pmemop() {
  return PyModule_Create(&pmemopmodule);
}
