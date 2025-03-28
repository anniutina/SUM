# [SUM](https://sum-project.eu/)

# Seamless Shared Urban Mobility (SUM)
> This software enables the generation of trip requests and the simulation of various travel scenarios within a defined study area. Moreover, it allows the assessment of key performance indicators (KPIs) for the New Shared Mode (NSM) integrated with public transport (PT).
## Technology
The software was implemented using Python programming language and its basic libraries
## Description
### The developed framework utilizes two main frameworks:
* [ExMAS](https://github.com/RafalKucharskiPK/ExMAS/tree/master/ExMAS)
* [Public Transport queries](https://github.com/RafalKucharskiPK/query_PT)
### The roadmap of the current project for the city:
1) Create a demand dataset. Sampling is done according to the established travel patterns. 
     
 ![OD](https://github.com/OlhaShulikaUJ/SUM_project/blob/main/OD.png) 

2) Calculate utilities of different transport modes:
   
  a) Utility of PT for trips from O to D: $U_{PT:O\to \overline{D}}$

  b) Utility of PT for trips from HUB to D: $U_{PT:HUB\to \overline{D}}$

  c) Utility of ride-pooling (following ExMAS) for all PT users from O to HUB:

$$
\begin{aligned}
 U_{SUM}=U_{PT:HUB\to \overline{D}} + \underbrace{\beta _{t}\beta _{s}\left ( t _{t}+ \beta _{w}t _{w}\right)}+ASC
\end{aligned}
$$

3) Mode Choice
   * сalculate ASC for the given $E(p_{sum})$
   * define the average ASC 
   * recalculate $p_{sum}$ for all PT users

4) Demand for SUM
   * sample travellers choosing SUM as: $p_{random} < p_{sum}$

5) KPIs for SUM
   * ExMAS-calculated KPIs for SUM users from O to HUB
   * assessment


## Input:
* [csv file](https://github.com/anniutina/SUM/blob/main/data/krk_demo_example.csv) with population and address points distribution 
  
* graphml file with [city graph](https://github.com/anniutina/SUM/blob/main/ExMAS/data/graphs/Krakow.graphml)
* the [default](https://github.com/anniutina/SUM/blob/main/data/configs/default_SUM.json) file with configurations
  
* dbf file with OSM network (available e.g. [here](https://www.interline.io/osm/extracts/))
* zip with GTFS file for the area and date that we query (available e.g. from [gtfs](https://gtfs.ztp.krakow.pl/))
* both OSM and GTFS files should be stored in the data folder

## Output:
* [csv](https://github.com/anniutina/SUM/tree/main/results) with results

## Usage:
* Simulations, KPI estimation and comparative analysis of two SUM areas [notebook](https://github.com/anniutina/SUM/blob/main/sum_main.ipynb)
* running the OTP server [notebook](https://github.com/OlhaShulikaUJ/SUM_project/blob/main/PT/run%20OTP%20server-KRK.ipynb)
