The dataset contains a gold standard for classifying parts of requirements in five traceability link recovery benchmark datasets.

# Classification

- For aspect classification:
    - functional aspects (F)
    - quality aspects (Q)
- For concerns in functional requirements (c.f. NoRBERT publication):
    - Function: A function that a system shall perform
    - Behavior: Behavior, the system displays or reactions that are triggered by one or more stimuli
    - Data: A data item or data structure that shall be part of a system's state
    - UserRelated: Behavior of the user or functionality of the system attributable to the user 
    
# Datasets

The dataset comprises preprocessed requirements of the eTour, iTrust, SMOS, eAnci and LibEST datasets. As SMOS and eAnci's original requirements were written in Italian, the dataset comprises automatically translated versions of the requirements to English. The datasets were retrieved from the website of the Center of Excellence for Software & Systems Traceability (CoEST). Attribution for the datasets:

The original eTour dataset was provided for the TEFSE challenge at 6th International Workshop on Traceability in Emerging Forms of Software Engineering (TEFSE), 2011 and was retrieved from http://coest.org/

The iTrust dataset was retrieved from http://coest.org/

The original SMOS and eAnci datasets can be attributed to Gethers et al., On integrating orthogonal information retrieval methods to improve traceability recovery. In 2011 27th IEEE International Conference on Software Maintenance (ICSM), Sep. 2011 and were retrieved from http://coest.org/ 

The LibEST dataset can be attributed to Moran et al., Improving the Effectiveness of Traceability Link Recovery using Hierarchical Bayesian Networks. In 2020 IEEE/ACM 42nd International Conference on Software Engineering (ICSE), May 2020 and was retrieved from https://gitlab.com/SEMERU-Code-Public/Data/icse20-comet-data-replication-package

