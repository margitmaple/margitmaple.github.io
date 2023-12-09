## Overview
In order to predict beach groundwater hydraulic head, an ensemble machine learning approach is appplied using linear regression with ridge regularization, random forrest decision trees, and artificial neural networks. 

***

## Introduction 
Coastal aquifers are highly dynamic groundwater systems which are influenced by coastal water levels.  Sea level fluctuations such as tides, storm surge and wave impacts propagate inland, causing the water table of adjacent unconfined aquifers (i.e., beach groundwater) to oscillate over the first hundred or so meters inland from their interface \cite{nielsen1990tidal, housego2021coastal}. Being able to characterize these fluctuations is crucial, as shallow coastal groundwater levels may result in a number of challenges including groundwater emergence and flooding \cite{befus2020increasing,habel2020sea}, contaminant mobilization \cite{kreibich2008assessment}, subsurface infrastructure damage \cite{habel2017development,knott2017assessing}, and increased liquefaction risk in seismically active regions \cite{abueladas2021liquefaction}. 

Although many analytical and numerical models have been used to explore these interactions, accurate prediction of beach water tables is limited due to the complex, varying topography and geology of coastal regions. Machine learning approaches provide an alternative approach that could allow the development of predictive models without requiring full knowledge of the field site. Recent studies have started to explore the application of machine learning techniques to groundwater forecasting (e.g. Bowes et al., 2019; Roshini et al., 2020); however, these studies look at timescales on the timescales of days to months, and therefore do not capture the shallow groundwater hazards associated with periodically elavated water levels. Additionally, wave conditions are not included in previous models. Along the U.S. West Coast, wave runup significantly increases maximum total coastal water levels \cite{serafin2017relative} and wave impacts are expected to intensify in a changing climate \cite{reguero2019recent, bromirski2023climate}.

This work aims to fill the gaps left by previous studies and assess the ability of supervised machine learning techniques to predict groundwater levels. Groundwater measurements and topographic surveys were conducted in the winter of 2014-2015 at Imperial Beach, CA. Along with publicly availble data of offshore conditions, this allows for the developement of an ensemble machine learning approach.

## Data
Input data features selected for this project are time (t), offshore water level (WL), deep water significant wave height (HSo), peak period (TP), peak direction (DP), foreshore beach slope (b). Off-shore wave conditions (HSo, Tp, and DP) at 30 minute intervals are publicly availible through the Coastal Data Information Program's (CDIP, https://cdip.ucsd.edu/) nearest deep water wave buoy (Point Loma South, 191). Water levels at 6 minute intervals are availible through the National Oceanic and Atmospheric Administration's (NOAA, https://tidesandcurrents.noaa.gov/) nearest open coast tide gauge (La Jolla, 9410230). 

Beach topography is taken from an ATV survey conducted February 18th, 2015. Foreshore beach slope varies as water levels move up and down the beach with tides and waves, as the beach face is convex. Therefore, an emperical parameterization of mean water level and wave runup presented by Stockdon et al., 2006 is used to determine the region over which the foreshore beach slope is recorded:

$`bound = 1.1\bigg(0.35b(H_{S0}L_0)^{\frac{1}{2}}\pm\dfrac{[H_{S0}L_0(0.563b^2+0.004)]^{\frac{1}{2}}}{2}\bigg) + WL`$

$`b=\dfrac{y(bound_{upper}) - y(bound_{lower})}{x_(bound{upper}) - x(bound_{lower})}`$

The groundwater head measured at the pressure sensor shown in Figure XXX is taken to be the target data. All input and targer data is interpolated to match the lowest sample rate of 30 minutes in the CDIP data, and normalized to range from 0 to 1.

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
