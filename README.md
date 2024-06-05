# [SUM](https://sum-project.eu/)

# Seamless Shared Urban Mobility (SUM)
> This software allows performing simulations of trip requests and various travel scenarios within the specified study area, and using obtained KPIs to determine utilities of the integration of New Shared Modes and PT.
## Technology
The software was implemented using Python programming language and its basic libraries
## Description
### The developed framework is based on two main developed libraries:
* [Public Transport queries](https://github.com/RafalKucharskiPK/query_PT)
* [ExMAS](https://github.com/RafalKucharskiPK/ExMAS/tree/master/ExMAS)
### A module 'The parameter-free comparison of selected areas for Krakow' consists of:
* [PT_mode](https://github.com/OlhaShulikaUJ/SUM_project/tree/main/PT)
* [NSM+PT_mode](https://github.com/OlhaShulikaUJ/SUM_project/tree/main/NSM%2BPT)
### The roadmap of the current project for the city:
1) OD
   * sample origins
   * sample destinations
     
 ![OD](https://github.com/OlhaShulikaUJ/SUM_project/blob/main/OD.png) 
     
2) Demand for PT
   
3) Utility of PT for trips from O to D: $U_{PT:O\to \overline{D}}$

4) Utility of PT for trips from HUB to D: $U_{PT:HUB\to \overline{D}}$

5) ExMAS for all PT users from O to HUB:

$$
\begin{aligned}
 U_{SUM}=U_{PT:HUB\to \overline{D}} + \underbrace{\beta _{t}\beta _{s}\left ( t _{t}+ \beta _{w}t _{w}\right)}+ASC
\end{aligned}
$$

6) Mode Choice
   * сalculate ASC for the given $E(p_{sum})$ for n replications
   * define the average ASC 
   * recalculate $p_{sum}$ for all PT users

7) Demand for SUM
   * count a number of travellers with probability $p_{random} < p_{sum}$

8) KPIs for SUM
   * ExMAS only for SUM users from O to HUB
   * assessment


## Input:
* [csv file]((https://github.com/anniutina/SUM/blob/main/data/krk_demographic.csv)) with demography and address points distribution 
  
* graphml file with [city graph](https://github.com/anniutina/SUM/blob/main/ExMAS/data/graphs/Krakow.graphml)
* the [default](https://github.com/anniutina/SUM/blob/main/data/configs/default_SUM.json) file with configurations
  
* dbf file with OSM network (available e.g. [here](https://www.interline.io/osm/extracts/))
* zip with GTFS file for the area and date that we query (available e.g. from [gtfs](https://gtfs.ztp.krakow.pl/))
* both OSM and GTFS files should be stored in the data folder

## Output:
* [csv](https://github.com/anniutina/SUM/tree/main/results) with results

## Usage:
* Utilities, Mode choice, Demand NSM definitions in this [notebook](https://github.com/anniutina/SUM/blob/main/sum_main.ipynb)
* running the OTP server [notebook](https://github.com/OlhaShulikaUJ/SUM_project/blob/main/PT/run%20OTP%20server-KRK.ipynb)
