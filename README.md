# Multimodal Framework for Federated Learning (MuFFLe)

## TODO List + References:
- [X] Reference previous years' repos as inspo for this GitHub repo
  - [GeneVar2](https://github.com/collaborativebioinformatics/GeneVar2), a little *too* good because it was a 2nd followup project
  - [Haploblock Clusters](https://github.com/collaborativebioinformatics/Haploblock_Clusters_ElixirBH25)
- [X] Stick w the default MIT License, just don't use GPL or LGPL because industry can't use it 
- [ ] Scaffold

![flowchart](figures/flowchart.png)

# Introduction (1 para)

# Methods (2 paras)


# How we built this tool

# How to use this tool

# Future Directions
There are some low-hanging fruit that this could be applied to. While searching for instances to create our proof-of-concept, we came across some data from the Real-time Analysis and Discovery in Integrated And Networked Technologies (RADIANT) group, which 
> seeks to develop an extensible, federated framework for rapid exchange of multimodal clinical and research data on behalf of accelerated discovery and patient impact. 
[RADIANT Public Data (AWS)](https://registry.opendata.aws/radiant/)

We elected not to use this dataset because the S3 bucket had "controlled access," which required filling out a form for approval and did not fit the fast-paced nature of the Hackathon. However, our federated learning framework could be easily extended to RADIANT's data, which contains
> Clinical data, Imaging data, Histology data, Genomic data, Proteomics data, and more [Children's Brain Tumor Network (CBTN)](https://cbtn.org/research-resources)

