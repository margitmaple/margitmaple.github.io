## Overview
In order to predict beach groundwater hydraulic head, an ensemble machine learning approach is appplied using linear regression with ridge regularization, random forrest decision trees, and artificial neural networks. 

***

## Introduction 
Coastal aquifers are highly dynamic groundwater systems which are influenced by coastal water levels.  Sea level fluctuations such as tides, storm surge and wave impacts propagate inland, causing the water table of adjacent unconfined aquifers (i.e., beach groundwater) to oscillate over the first hundred or so meters inland from their interface \cite{nielsen1990tidal, housego2021coastal}. Being able to characterize these fluctuations is crucial, as shallow coastal groundwater levels may result in a number of challenges including groundwater emergence and flooding \cite{befus2020increasing,habel2020sea}, contaminant mobilization \cite{kreibich2008assessment}, subsurface infrastructure damage \cite{habel2017development,knott2017assessing}, and increased liquefaction risk in seismically active regions \cite{abueladas2021liquefaction}. 

Although many analytical and numerical models have been used to explore these interactions, accurate prediction of beach water tables is limited due to the complex, varying topography and geology of coastal regions. Machine learning approaches provide an alternative approach that could allow the development of predictive models without requiring full knowledge of the field site. Recent studies have started to explore the application of machine learning techniques to groundwater forecasting (e.g. Bowes et al., 2019; Roshini et al., 2020); however, these studies look at timescales on the timescales of days to months, and therefore do not capture the shallow groundwater hazards associated with periodically elavated water levels. Additionally, wave conditions are not included in previous models. Along the U.S. West Coast, wave runup significantly increases maximum total coastal water levels \cite{serafin2017relative} and wave impacts are expected to intensify in a changing climate \cite{reguero2019recent, bromirski2023climate}.

This work aims to fill the gaps left by previous studies and assess the ability of supervised machine learning techniques to predict groundwater levels. Groundwater measurements and topographic surveys were conducted in the winter of 2014-2015 at Imperial Beach, CA. Along with publicly availble data of offshore conditions, this allows for the developement of an ensemble machine learning approach.

Historical and forecasted off-shore wave conditions are publicly availible through the Coastal Data Information Program (CDIP, https://cdip.ucsd.edu/), and water level data is availible through the National Oceanic and Atmospheric Administration (NOAA, https://tidesandcurrents.noaa.gov/)
## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)
