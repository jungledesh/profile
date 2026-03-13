## Ground-Truth Metrics

Formulas come from theory, research, and actual usage.

---

### I. Hardware Primitives (The Physics)

- **GPU_ID**: Device identifier.  
- **HBM_Used**: Total VRAM usage.  
- **HBM_Throughput**: Memory bandwidth speed.  
- **SM_Util**: Core compute utilization.  
- **Power_W**: Energy consumption.  
- **Clock_Freq**: Throttling detection.  
- **HW_Timestamp**: High-precision temporal sync.  
- **Interconnect_Throughput**: PCIe/NVLink data movement.  
- **Mem_Stall_Cycles**: Detection of the “Memory Wall.”  

---

### II. Software: Lifecycle & Infrastructure

- **Req_ID**: Request identification for attribution.  
- **Phase_Flag**: Prefill vs. decode state.  
- **Block_Map**: KV cache paging and fragmentation.  
- **Cache_Metadata**: Prefix hashing for reuse rates.  
- **Arrival_TS**: User UX baseline.  
- **Scheduled_TS**: Scheduler lag entry point.  
- **Execution_Start_TS**: When the GPU kernels actually begin.  
- **Queue_Snapshot**: Total pending requests in the system.  
- **Tenant_ID**: Multi-tenancy and fairness tracking.  
- **Error_Event**: Timeouts, drops, and 5xx failures.  

---

### III. Software: Batch & Request Properties (NEW)

- **Active_Batch_Size** (NEW): The exact number of requests sharing a kernel execution.  
  - Solves: batch collapse (GPU idle because batch size is 1) and batch utilization (actual vs. max).  
- **Prompt_Token_Count** (NEW): The explicit size of the input.  
  - Solves: prefill efficiency. Lets you say: “This 1000-token prompt cost X joules to process.”  
- **Total_Output_Token_Count** (NEW): The final count of tokens generated.  
  - Solves: token length variance and total decode efficiency.  
- **Token_Event**: The live stream delta (Δt) for real-time TPOT.  
- **Token_Type / Static_Config**: Classification (speculative vs. target) and model ceilings.  