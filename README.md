# CUDA Toolkit Installation

## Prerequisites

Before jumping to CUDA installation, you need to check some information about your videocard:

1. Check whether your videocard supports CUDA [here](https://en.wikipedia.org/wiki/CUDA#:~:text=CUDA%20works%20with%20all%20Nvidia,with%20most%20standard%20operating%20systems.).
2. Get information about your videocard's `Microarchitecture`, `Compute Capability` and `Shader Model` using the same link.

Example:

```
Videocard:              NVIDIA GeForce GTX 960M
Microarchitecture:      Maxwell
Compute Capability:     5.0
Shader Model:           5.0
```

## Installing CUDA (Windows)

1. Go to [NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit) and choose the SDK Toolkit that corresponds to your videocard's `Compute Capability` and `Microarchitecture`. (List of [SDK Toolkits](https://en.wikipedia.org/wiki/CUDA#:~:text=CUDA%20works%20with%20all%20Nvidia,with%20most%20standard%20operating%20systems.))
2. Install and unpack the SDK.
3. Follow the prompts and in `Installation Options` choose `Express` option.
4. Wait until the process is finished and now you are done with CUDA. 

## Installing the IDE (Windows)

1. Install Microsoft VisualStudio (Recommended).
2. Open the VS Installer and check the `C++ Build Tools` workload and be sure to check `MSVS vertion 140` or `MSVS vertion 141` (varies for different SDKs). 
3. Click Install.

## Setting the IDE up

1. Create a `CUDA Project`.
2. Go to ` Project -> Properties -> CUDA C/C++ -> Device `.
3. In field `Code Generation` enter `compute_<compute_capability>, sm_<shader_model>`.
4. Click `Apply`.

Now you are ready to go!
