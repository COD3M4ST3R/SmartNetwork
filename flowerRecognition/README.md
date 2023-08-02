
# Greetings, Welcome To 'image_recognition'!
This project has been made to teach people who wants to learn or improve themselfs in the field of AI.

## What Is This Project?
This is a supervised image recognition project that has been completed with 'Tensorflow' & 'Keras' as highlighted technologies.

## What Its Main Purpose?
Teaching & improvement. There are many sources that you can examine and learn AI from them but since this field is improving with a speed of light, sources might get outdated. 

Many of the AI projects to teach basics has been made as small scale project therefore they are easy to understand at first glance but extremely hard to classify into bigger scaled projects. With this project, you will face with well structured architecture for bigger scaled projects. 

At first, it might be little bit excessive and hard to understand, but when you overcome the learning curve, you will love it and will be able to clearly observe the architecure choices that has been made makes your job easier to understand what exactly accomplishes what.

## What Makes This Project Different Than Others?
It is made with known technologies and achive relatively easy task to accomplish. What makes difference is that structure designed very well and we both know how 'Tensorflow' can be tough to understand since it is not 'pythonic'. And well you know, documentations from Google are not helping much either.

Every macro and parameters for AI (Hyperparameters, Models, Optimizers, Load & Preprocess, Input & Output tuning, Future Engineering, Datasets, Activation Functions, etc...) has been defined inside parameter file called 'values.yaml' therefore you can change everything inside the programm from one file, even the dataset, and yes I approached with a 'Repository' design for this project. 

You will able to change everything to observe which model works best with what and when. This will increase your understanding and classification towards processes.

With custom created input pipeline of 'Load & Preprocess' process, you will be able to perform 'Data Engineering' over your project to improve outcomes.

# License
**MIT LICENSE**
Please feel free to change everything to improve anything. Is there something I can do better? Let me know. 

# Learning Path For AI
```mermaid
flowchart TD
    WhatIsAnAi([What Is An AI?]) -->|Learn both technically and philosophically| WhatWeCanAchieveWithAnAi[What We Can Achieve With An AI?]
    
    WhatWeCanAchieveWithAnAi --> |Examine completed projects to observe horizon| HowCanICompleteAnAIProject[How Can I Complete an AI Project?]
    
    HowCanICompleteAnAIProject -->|Determine a project that you would like to complete| WhatWillBeTheRequirements[What Will Be The Requirements?]

    WhatWillBeTheRequirements --> |Decide which programming language fits into your needs| ProgrammingLanguages(Programming Languages)

    WhatWillBeTheRequirements --> |Learn concepts and needs| DataAndDataEngineering(Data And Data Engineering)

    WhatWillBeTheRequirements --> |Observe differences and understand their advantages with disadvantages| FrameworksAndLibraries(Frameworks And Libraries)

    WhatWillBeTheRequirements --> |Learn domains in order to be able to differentiate what really matters|FeatureEngineering(Feature Engineering)

    WhatWillBeTheRequirements --> |Learn how to analyze results to improve them|Analysis(Analysis)
    
    WhatWillBeTheRequirements --> |Check what are those Hyperparameters and how they are effecting the results|Hyperparameters(Hyperparameters)
    
    Analysis --> |In order to analysis data, you need to know how to visualize data for more efficiency| Visualization(Visualization)

    Analysis --> |Need to gather enough data from trained model to understand and analysis the performance of the model| Performance(Performance)
    
    Performance --> Overfitting(Overfitting)

    Performance --> Underfitting(Underfitting)

    Performance --> Appropriatefitting(Appropriatefitting)

    ProgrammingLanguages --> LearnImproveKnowledge[/Learn / Improve Knowledge/]

    LearnImproveKnowledge --> |Learn what algorithms differentiates AI from traditionals programmes|Algorithms(Algorithms)

    DataAndDataEngineering --> |Learn how to collect relavent data| DataCollection(Data Collection)

    DataAndDataEngineering --> |Learn how to store collected data| DataStorage(Data Storage)

    Pipelines --> |Learn how to deal with high volume of data in efficient ways| Database[(Database)]

    DataCollection --> Pipelines(Pipelines)

    DataStorage --> Pipelines

    Pipelines --> |Learn how to load and preprocess data|LoadAndPreprocess(Load And Preprocess)

    LoadAndPreprocess --> DataAugmentation(DataAugmentation)

    Algorithms --> Models(Models)

    Algorithms --> |Learn the types of Activation Functions and their mentality|ActivationFunctions(Activation Functions)

    Algorithms --> |Learn the types of Optimizers and their mentality|Optimizers(Optimizers)

    Models --> |Learn the types of ANNs and their mentality| ANNs(ANNs)

    Models --> RegressionAndClassificationAlgorithms(Regression And Classification Algorithms)

    ANNs --> ANNProcessing(ANN Processing)

    ANNProcessing --> BackPropagation[/Back Propagation/]

    ANNProcessing --> Forwarding[/Forwarding/]

    FrameworksAndLibraries --> Tensorflow[/Tensorflow/]

    FrameworksAndLibraries --> PyTorch[/PyTorch/]

    FrameworksAndLibraries --> Jax[/Jax/]

    FrameworksAndLibraries --> CNTK[/CNTK/]

    Visualization --> Matplotlib[/Matplotlib/]
```