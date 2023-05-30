#include <thread>
#include <pthread.h>
#include <mutex>
#include <queue>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <condition_variable>
#include "ready_event.h"
#include <stdio.h>
#include <regex.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef PMEMOP_OBJECT_H
#define PMEMOP_OBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

struct ChkRequest {
    int req_type;
    PyObject *dict = NULL;
    PyObject *key = NULL;
    PyObject *value = NULL;
    int handle = 0;

    ChkRequest(int req_type, PyObject *dict, PyObject *key, PyObject *value, int handle);
};

struct ChkGlobalState {
    PyObject *pm_dict[2];
    PyObject *dram_dict[2];
    PyObject *model_dict[2];
    PyObject *opt_dict[2];
    PyObject *opt_state_dict[2];
    std::thread chk_thread;
    std::mutex que_mutex;
    std::condition_variable que_cond;
    int chk_flag;
    int save_flag;
    std::mutex save_mutex;
    uint64_t handle = 0;
    std::unordered_map<int, std::shared_ptr<ReadyEvent>> ready_events;
    std::unordered_map<int, int> ready_save_flags;
    std::mutex event_mutex;
    std::atomic_bool initial_done;
    std::queue<std::shared_ptr<struct ChkRequest>> chk_que;
    int tensor_copy_num;
    int tensor_num;
    int copy_flag;
};

void set_chk_way(long way);
PyObject *PM_PyDict_Write(PyObject *o);
PyObject *PM_PyDict_New(void);
Py_ssize_t All_PyDict_KeysSize(PyDictKeysObject *keys);
int PM_PyDict_SetItem(ChkGlobalState& chk_state, PyObject *dict, std::shared_ptr<struct ChkRequest> req);
void PM_PyTensor_SetItem(ChkGlobalState& chk_state, int handle, PyObject *pm_value, PyObject *dram_value);
PyObject *PM_PyList_Write(PyObject *a);
void PM_PyDict_Copy(ChkGlobalState& chk_state, PyObject *dst, PyObject *src);
void PM_PyList_Copy(ChkGlobalState& chk_state, PyObject *b, PyObject *a);
PyObject *PM_PyList_New(Py_ssize_t size);
PyObject *PM_PyTuple_Write(PyObject *aa);
void PM_PyTuple_Copy(ChkGlobalState& chk_state, PyObject *b, PyObject *a);
PyObject *PM_PyTuple_New(Py_ssize_t size);
PyObject *PM_PyFloat_FromDouble_Write(double fval);
void PM_PyFloat_FromDouble_Copy(PyObject *dst, double fval);
PyObject *PM_PyLong_Write(PyObject *value);
void PM_PyLong_Copy(PyObject*dst, PyObject *value);
PyLongObject *PM_PyLong_New(Py_ssize_t size);
PyObject *PM_PyUnicode_Write(PyObject *unicode);
void PM_PyUnicode_Copy(PyObject *dst, PyObject *unicode);
PyObject *PM_PyUnicode_New(Py_ssize_t size, Py_UCS4 maxchar);
void ChkThreadLoop(ChkGlobalState& chk_state);
bool RunLoop(ChkGlobalState& chk_state);
void write_tensor_data(PyObject *pm_dict, PyObject *chk_dict, PyObject *buffer_name_list, PyObject *mapping);
void PM_PyTensor_Copy(ChkGlobalState& chk_state, PyObject *pm_value, PyObject *dram_value);
PyObject *PM_PyTensor_Write(PyObject *value);
bool poll_all_events(ChkGlobalState& chk_state);
bool poll_handle(ChkGlobalState& chk_state, int handle);
PyObject *PM_PyLong_FromVoidPtr(void *ptr);
PyObject *PM_PyLong_FromLongLong(long long ival);
void PM_PyDict_Read(PyObject *dram_dict, PyObject *pm_dict);
void PM_PyList_Read(PyObject *dram_list, PyObject *pm_list);
void PM_PyTuple_Read(PyObject *dram_list, PyObject *pm_list);
void PM_PyTensor_Read(PyObject *dram_value, PyObject *pm_value);


struct DimmAttribute128b {
	uint64_t l_u64b, h_u64b;
};

struct DimmDataContainer {
	char dimm_id_[100];
	struct DimmAttribute128b stat_[8];
};

struct PMMData {
	struct DimmDataContainer pmm_dimms_[16];
	int dimm_size;
    char *name;
};


enum
{
    DimmID = 0,
    MediaReads,
    MediaWrites,
    ReadRequests,
    WriteRequests,
    TotalMediaReads,
    TotalMediaWrites,
    TotalReadRequests,
    TotalWriteRequests
};


static void get_pmm_data(struct PMMData *pd)
{
    char file[100];
    char cmd[100];
    memset(file, 0, sizeof(file));
    memset(cmd, 0, sizeof(cmd));
    sprintf(file, "pmm_stat%s.txt", pd->name);
    sprintf(cmd, "ipmctl show -performance > %s", file);
	if (system(cmd) == -1) {
        printf("system call fail\n");
        exit(1);
    }
	FILE *ipmctl_stat = NULL;
    
	ipmctl_stat = fopen(file, "r"); // open the input file
	if (ipmctl_stat == NULL) {
		printf("open file pmm_stat.txt error\n");
		return;
	}

	char reg_init_set[][50] = {
		R"(DimmID=0x([0-9a-f]*))",
        R"(^\s+MediaReads=0x([0-9a-f]*))",
        R"(^\s+MediaWrites=0x([0-9a-f]*))",
        R"(^\s+ReadRequests=0x([0-9a-f]*))",
        R"(^\s+WriteRequests=0x([0-9a-f]*))",
        R"(^\s+TotalMediaReads=0x([0-9a-f]*))",
        R"(^\s+TotalMediaWrites=0x([0-9a-f]*))",
        R"(^\s+TotalReadRequests=0x([0-9a-f]*))",
        R"(^\s+TotalWriteRequests=0x([0-9a-f]*))"};
    int reg_size = sizeof(reg_init_set) / sizeof(reg_init_set[0]);

    regex_t stat_bit_convert_reg;
    char stat_bit_convert_str[] = (R"(^([0-9a-f]{16})([0-9a-f]{16}))");
    regex_t reg_set[10];
    for (int i = 0; i < reg_size; i++) {
    	regcomp(&reg_set[i], reg_init_set[i], REG_EXTENDED | REG_NEWLINE);
    }
    regcomp(&stat_bit_convert_reg, stat_bit_convert_str, REG_EXTENDED | REG_NEWLINE);

    regmatch_t pmatch1[2];
    char matched_buf[100];
    regmatch_t pmatch2[3];
    char matched_num[100];
    char str_line[100];
    int index;
    while (fgets(str_line, 100, ipmctl_stat) != NULL) {
        // printf("%s", str_line);
    	for (index = 0; index < reg_size; index++) {
    		int status = regexec(&reg_set[index], str_line, 2, pmatch1, 0);
    		if (status == 0) {		// match
                // printf("match at idx %d\n", index);
    			break;
    		}
    	}
        if (index >= reg_size) {
            printf("parse dimm not match 1\n");
            exit(1);
        }
        memset(matched_buf, 0, sizeof(matched_buf));
    	memcpy(matched_buf, str_line + pmatch1[1].rm_so, pmatch1[1].rm_eo - pmatch1[1].rm_so);
    	if (index == DimmID) {
            pd->dimm_size++;
            memset(pd->pmm_dimms_[pd->dimm_size - 1].dimm_id_, 0, sizeof(pd->pmm_dimms_[pd->dimm_size].dimm_id_));
    		memcpy(pd->pmm_dimms_[pd->dimm_size - 1].dimm_id_, matched_buf, strlen(matched_buf));
            // printf("DimmID: %s\n", pd->pmm_dimms_[pd->dimm_size - 1].dimm_id_);
    	}
        else {
            int status = regexec(&stat_bit_convert_reg, matched_buf, 3, pmatch2, 0);
            if (status != 0) {
                printf("parse dimm not match 2\n");
                exit(1);
            }
            memset(matched_num, 0, sizeof(matched_num));
            memcpy(matched_num, matched_buf + pmatch2[1].rm_so, pmatch2[1].rm_eo - pmatch2[1].rm_so);
            pd->pmm_dimms_[pd->dimm_size - 1].stat_[index - 1].h_u64b = strtoull(matched_num, NULL, 16);
            memset(matched_num, 0, sizeof(matched_num));
            memcpy(matched_num, matched_buf + pmatch2[2].rm_so, pmatch2[2].rm_eo - pmatch2[2].rm_so);
            pd->pmm_dimms_[pd->dimm_size - 1].stat_[index - 1].l_u64b = strtoull(matched_num, NULL, 16);
            // printf("%lld\n", pd->pmm_dimms_[pd->dimm_size - 1].stat_[index - 1].l_u64b);
        }
        memset(str_line, 0, sizeof(str_line));
    }

    for (int i = 0; i < reg_size; i++) {
        regfree(&reg_set[i]);
    }
    regfree(&stat_bit_convert_reg);
}

static struct PMMData *start = NULL;
static struct PMMData *end = NULL;
static int print_flag_ = 1;

static struct timespec start_timer, end_timer;
static float *outer_imc_read_addr_ = NULL, *outer_imc_write_addr_ = NULL,
      *outer_media_read_addr_ = NULL, *outer_media_write_addr_ = NULL;

static void PmmDataCollector(const char *name, float *real_imc_read, float *real_imc_write,
                      float *real_media_read, float *real_media_write) {
    printf("start pmmdata collector\n");
    outer_imc_read_addr_ = real_imc_read;
    *outer_imc_read_addr_ = 0;
    outer_imc_write_addr_ = real_imc_write;
    *outer_imc_write_addr_ = 0;
    outer_media_read_addr_ = real_media_read;
    *outer_media_read_addr_ = 0;
    outer_media_write_addr_ = real_media_write;
    *outer_media_write_addr_ = 0;
    clock_gettime(CLOCK_REALTIME, &start_timer);
    start = (struct PMMData*)malloc(sizeof(struct PMMData));
    start->dimm_size = 0;
    start->name = "start";
    get_pmm_data(start);
}

static void PmmDataFinish() {
    printf("start compute pmmdata\n");
    end = (struct PMMData*)malloc(sizeof(struct PMMData));
    end->dimm_size = 0;
    end->name = "end";
    get_pmm_data(end);
    clock_gettime(CLOCK_REALTIME, &end_timer);
    float media_read_size_MB = 0, imc_read_size_MB = 0, imc_write_size_MB = 0, media_write_size_MB = 0;
    double sec = (end_timer.tv_sec - start_timer.tv_sec) + (end_timer.tv_nsec - start_timer.tv_nsec) / 1000000000.0;
    if (print_flag_) {
        fprintf(stderr, "-------------------------------------------------------------------------\n");
        fprintf(stderr, "elapsed time: %'.2f sec\n", sec);
        fprintf(stderr, "-------------------------------------------------------------------------\n");
	fprintf(stderr, "|DIMM\t|RA\t|WA\t|iMC Rd(MB)\t|Media Rd(MB)\t|iMC Wr(MB)\t|Media Wr(MB)\t|Num Rd Rq\t|Num Wr Rq\t|Num Rd Md\t|Num Wr Md\t|\n");
    }
    float TotalMediaReads_[16], TotalMediaWrites_[16], TotalReadRequests_[16], TotalWriteRequests_[16];
    uint64_t numReadRequests_[16], numWriteRequests_[16], numMediaReads_[16], numMediaWrites_[16];
    uint64_t totalReadNum = 0, totalWriteNum = 0, totalMediaReadNum = 0, totalMediaWriteNum = 0;
    for (int i = 0; i < end->dimm_size; i++) {
        TotalMediaReads_[i] = (end->pmm_dimms_[i].stat_[TotalMediaReads - 1].l_u64b - start->pmm_dimms_[i].stat_[TotalMediaReads - 1].l_u64b) / 16384.0;
        TotalMediaWrites_[i] = (end->pmm_dimms_[i].stat_[TotalMediaWrites - 1].l_u64b - start->pmm_dimms_[i].stat_[TotalMediaWrites - 1].l_u64b) / 16384.0;
        TotalReadRequests_[i] = (end->pmm_dimms_[i].stat_[TotalReadRequests - 1].l_u64b - start->pmm_dimms_[i].stat_[TotalReadRequests - 1].l_u64b) / 16384.0;
        TotalWriteRequests_[i] = (end->pmm_dimms_[i].stat_[TotalWriteRequests - 1].l_u64b - start->pmm_dimms_[i].stat_[TotalWriteRequests - 1].l_u64b) / 16384.0;

	numMediaReads_[i] = (end->pmm_dimms_[i].stat_[TotalMediaReads - 1].l_u64b - start->pmm_dimms_[i].stat_[TotalMediaReads - 1].l_u64b);
        numMediaWrites_[i] = (end->pmm_dimms_[i].stat_[TotalMediaWrites - 1].l_u64b - start->pmm_dimms_[i].stat_[TotalMediaWrites - 1].l_u64b);
        numReadRequests_[i] = (end->pmm_dimms_[i].stat_[TotalReadRequests - 1].l_u64b - start->pmm_dimms_[i].stat_[TotalReadRequests - 1].l_u64b);
        numWriteRequests_[i] = (end->pmm_dimms_[i].stat_[TotalWriteRequests - 1].l_u64b - start->pmm_dimms_[i].stat_[TotalWriteRequests - 1].l_u64b);

        TotalMediaReads_[i] = TotalMediaReads_[i] - TotalMediaWrites_[i];
	numMediaReads_[i] = numMediaReads_[i] - numMediaWrites_[i];

        imc_read_size_MB += TotalReadRequests_[i];
        media_read_size_MB += TotalMediaReads_[i];
        imc_write_size_MB += TotalWriteRequests_[i];
        media_write_size_MB += TotalMediaWrites_[i];

	totalReadNum += numReadRequests_[i];
	totalWriteNum += numWriteRequests_[i];
	totalMediaReadNum += numMediaReads_[i];
	totalMediaWriteNum += numMediaWrites_[i];
    }
    if (outer_imc_read_addr_)
        *outer_imc_read_addr_ = imc_read_size_MB;
    if (outer_imc_write_addr_)
        *outer_imc_write_addr_ = imc_write_size_MB;
    if (outer_media_read_addr_)
        *outer_media_read_addr_ = media_read_size_MB;
    if (outer_media_write_addr_)
        *outer_media_write_addr_ = media_write_size_MB;
    if (print_flag_) {
        for (int i = 0; i < end->dimm_size; i++) {
            fprintf(stderr, "|0x%s\t", start->pmm_dimms_[i].dimm_id_);

            if ((TotalMediaReads_[i] / TotalReadRequests_[i] > 5))
                fprintf(stderr, "|N/A\t");
            else
                fprintf(stderr, "|%'.2f\t", TotalMediaReads_[i] / TotalReadRequests_[i]);
            if ((TotalMediaWrites_[i] / TotalWriteRequests_[i]) > 5)
                fprintf(stderr, "|N/A\t");
            else
                fprintf(stderr, "|%'.2f\t", TotalMediaWrites_[i] / TotalWriteRequests_[i]);
            fprintf(stderr, "|%'8.2f\t", TotalReadRequests_[i]);
            fprintf(stderr, "|%'8.2f\t", TotalMediaReads_[i]);
            fprintf(stderr, "|%'8.2f\t", TotalWriteRequests_[i]);
            fprintf(stderr, "|%'8.2f\t", TotalMediaWrites_[i]);

            fprintf(stderr, "|%10lu\t", numReadRequests_[i]);
            fprintf(stderr, "|%10lu\t", numWriteRequests_[i]);
            fprintf(stderr, "|%10lu\t", numMediaReads_[i]);
            fprintf(stderr, "|%10lu\t|\n", numMediaWrites_[i]);
        }
        fprintf(stderr, "\033[32mTotal RA: %f, iMC read %fMB, media read %fMB, read request num %lu, media read num %lu\033[0m\n", media_read_size_MB / imc_read_size_MB, imc_read_size_MB, media_read_size_MB, totalReadNum, totalMediaReadNum);
        fprintf(stderr, "\033[31mTotal WA: %f, iMC write %fMB, media write %fMB, write request num %lu, media write num %lu\033[0m\n", media_write_size_MB / imc_write_size_MB, imc_write_size_MB, media_write_size_MB, totalWriteNum, totalMediaWriteNum);
    }
    free(start);
    free(end);
}

#ifdef __cplusplus
}
#endif
#endif /* !PMEMOP_OBJECT_H */
