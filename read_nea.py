# -*- coding: utf-8 -*-
"""
read_nea
 (1) READ the data from NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu/)
 (2) MAKE .csv file

@author: wskang
@update: 2022/01/26
"""
import numpy as np
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

nea_table = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="pl_name, pl_orbper, st_rad, pl_orbsmax, pl_radj, pl_imppar",
    where=f"pl_orbper > 0 and pl_orbsmax > 0")

f = open('nea.csv','w')
for line in nea_table:
    PNAME = line['pl_name'].replace(' ','')
    PER = line['pl_orbper'].value
    A = line['pl_orbsmax'].value
    RSTAR = line['st_rad'].value
    RPLANET = line['pl_radj'].value
    RR = (RPLANET / RSTAR) * 0.10045
    B = line['pl_imppar']
    fstr = f"{PNAME}, {PER}, {RSTAR}, {A}, {RR}, {B}"
    f.write(fstr+"\n")
    print(fstr)

f.close()
