#!/bin/bash
declare -a hash=("#BreatheLife" "#AirPollution" "#airquality" "#cleanair" "#airpollution" "#pollution" "#hvac"
            "#airpurifier" "#indoorairquality" "#air" "#climatechange" "#indoorair" "#environment"
            "#airconditioning" "#heating" "#freshair" "#airfilter" "#ventilation" "#airconditioner"
            "#airqualityindex" "#pm2_5 " "#emissions" "#natureishealing" "#nature" "#pollutionfree"
            "#wearethevirus" 'AirPollution' 'Environment' 'Ozone Layer' 'Global Warming' 'Climate Change'
            'Greenhouse Gases' 'Trees' 'Carbon' 'Aerosals' 'Air' 'Save the planet' 'Factories' 'Hygroscopicity'
            'Inversion' 'Sulfur' 'AIRS' 'ecosystem' 'Hydrochlorofluorocarbon' 'hydrocarbon' 'TAC' 'zero'
            'pollutant' '#air' '#pollution' '#airpollution' '#coal' '#particles' '#smog' '#cleanair'
            '#airqualityindex' '#climatechange' '#airquality' '#globalwarming' '#airpollutionawareness'
            '#airpollutioncontrol' '#CleanEnergy' '#saveearth')

declare -a geo=("6.48937,3.37709,500km" "-33.99268,18.46654,500km" "-26.22081,28.03239,500km" "5.58445,-0.20514,500km"
            "-1.27467,36.81178,500km" "-4.04549,39.66644,500km" "-1.95360,30.09186,500km" "0.32400,32.58662,500km")

for i in "${hash[@]}"
do
    for j in "${geo[@]}"
    do
        snscrape twitter-search "$hash geocode:$j since:2020-04-01 until:2020-07-07" > "$i$j.txt"
    done
done