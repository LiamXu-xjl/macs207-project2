# Basic Information
This is the GitHub Repo for our project 2 of CWL/MACS 207 Spring 2021 at University of Illinois at Urbana Champaign

The formal course description and other informations can also be found [here](https://courses.illinois.edu/schedule/2021/spring/MACS/207)

# Team members: 

Jialiang Xu (jx17)

Jiayuan Hu (jiayuan4)

Yajing Gao (yajingg2)

Yizhen Lu (yizhenl3)

# Usage
```
pip install requirements.txt
jupyter notebook KE\ movie\ plot.ipynb
```

# Introduction 

One of the vital elements of a movie is the plot. More often than not, the plot provides most intuitive presentations of the underlying movie logic and carries most information facilitating viewer understanding. Thus, the task of **plot understanding** is one of the most valuable tasks in movie analysis.  

Currenlty, it is very common that **a major portion of effort and time is paid to extract the plot features when conducting movie analysis.** Particularly, when the analysis involves multiple movies, manual approaches takes even longer time as per the extra counting and statistics steps involved. 

This project aims to **offer a novel perspective to the task of plot understanding in the context of modern Indian Cinema.** We tackle the problem mainly with Natural Language Processing techniques. We propose an automated pipeline to extract the plot features and generate latent representations for the features. We also included visualization modules to convert the latent features extracted to existent keywords in the plot, and present them in the form of wordclouds. 

The extracted plot features can then be used in downstream tasks such as 

1. Automated movie recommendation
2. Similarity examination
3. Movie classification
4. ... 

As a proof of concept, we selected the top 10 highest grossing movies in the domestic theatre market in India and feed them into the pipeline. **We aim to find similar elements in the plots of these movies.** Via this approach, we expect to gain knowledge about the viewership, specifically, their preferences for plots.    

To refrain the range of the task, we made the following assumptions: 
1. We regard it a reasonable assumption that the response of a large crowd in a society towards a specific movie is predominantly controlled by the plot, i.e. the plot of a movie explains the primary component of its box office collection. 
2. We further assume that the viewer preference is the only factor that determines movie box office collection. 



To summarize, in this project, we

1. Propose an automated pipeline that is capable of extracting latent features in movie plots and quantitatively compare them. 
2. Propose visualizations of the latent plot features with keywords existing in the movie plots.
3. Provide a proof of concept instance of extracting the common features of the top 10 highest grossing movies in the domestic theatre market in India. 


# Method

## Plot accessing 

We access the plot of target movies with the WikiPedia APIs. We use the raw text in the "slot" section on the WikiPedia page of the movie as our plot text. 

## Keyword Extraction 
### Pretrained models (BERT)
### EmbedRank and Maximum Marginal Relevance

## Visual representaion
### Wordcloud
### Mask


