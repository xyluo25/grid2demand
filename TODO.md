Actions:

---
### Overall link: https://www.caliper.com/tctraveldemand.htm
### Good source: https://github.com/joshchea/python-tdm/tree/master
2. [ ] Add load_zone_census_tracts
3. [ ] Trip generateion (Production and Attraction): 1) corss-classification method  2) Regression Methods  3) Discrete-Choice Methods
4. [ ] Mode choice: multinomial logit model, Nested logit model
    src: https://github.com/linhx25/MNLogit-zoo/tree/main ; https://github.com/ryonsd/choice-modeling;
5. [ ] Traffic assignment: All-or-nothing; STOCH ; Incremental assignment; Capacity Restraint; User  Equilibrium(Frank-Wolf method); Stochastic User Equilibrium (Method of Successive Averages, Sheffi and Powell, 1982; Sheffi, 1985); System optimum Assignment (SO);
* [ ] The calculated demands value are small -> change to 'reasonable' values
7. [ ] Advanced Traffic Assignment: Alternative or user-defined volume delay function; HOV assignment; Multi-modal multi-class assignment(MMA);  Volume-Dependent Turning delays and signal opeimization traffic assignment; Combined trip distribution- assignment model; Create volume delay function DDLs

9. [ ] Link performance functions  t = $t_f [1 + \alpha(\dfrac{v}{c})^\beta)] $
10. [ ] Enable multiple resources: osm2gmns, osmnx, overture, US Census Data...
* [X] Production and Attraction calculation: if poi exits, calculate poi based demand. elif activity node exist, calculate activity_node based demand, else, 0
* [X] trip distribution: gravity model
* [X] Add save node and poi function
* [X] User can update trip rate
* [X] For zone file, change id to zone_id
* [X] Add multiprocessing on zone centroid mapping with node and poi
