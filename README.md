# MicroIRC

## MicroIRC: Instance-level Root Cause Localization for Microservice Systems

  MicroIRC is an instance-level root cause localization method for microservice systems.
  The utilization of microservice architecture is gaining popularity in the development of Web applications. However, identifying the root cause of a failure can be challenging due to the complexity of interconnected microservices, long service invocation links, dynamic changes in service states, and the abundance of service deployment nodes. Furthermore, as each microservice may have multiple instances, it can be difficult to identify the instance-level failure promptly and effectively when microservice topologies and failure types change dynamically. To address this issue, we propose MicroIRC (Instance-level Root Cause Localization for Microservices), a novel metrics-based approach that localizes root causes at the instance level while exhibiting robustness to adapt to new types of anomalies. We begin by training a graph neural network to fit different root cause types based on extracted time series features of microservice system metrics. Next, we construct a heterogeneous weighted topology (HWT) of microservice systems and execute a personalized random walk to identify root cause candidates. These candidates and real-time metrics within the anomaly time window are then fed into the original graph neural network, generating a ranked root cause list. Remarkably, it exhibits robustness in scenarios involving multiple instances and new failure types.

## Framework

<img width="1276" alt="image" src="https://github.com/WHU-AISE/MicroIRC/assets/48899336/98e26e68-dd65-4373-970a-8a340ab617ac">

## Folder Introduction

### data

contains dataset *C* in the main branch and dataset *E* in the topoChange branch.

### model

store MetricSage multi-level parameter settings' models

### metric_sage

code of MetricSage GCN

## Getting Started

### Environment

```
python 3.10
```

### Clone the Repo

```shell
git clone https://github.com/WHU-AISE/MicroIRC.git
```

### TopoChange

The branch topoChange contains the extension of MetricSage for dynamic changes in topology and dataset *E*.

```shell
git checkout topoChange
```

### Install Dependencies

```shell
pip install -r requirements.txt
```

### Run

```shell
python MicroIRC.py
```
