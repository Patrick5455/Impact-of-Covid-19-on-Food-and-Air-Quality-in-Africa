import pandas as pd

df = pd.read_csv('totalDatanew_users_detail.csv')
df.hashtags.dropna(inplace=True)

hash3 = ["#AirPollution", '#stopairpollution', '#NO2Coal', '#EndCoal', '#AQI', '#airqualityindex', "#airquality",
         "#cleanair", "#airpollutant", "#freshair", "Particulates", "#pm2_5", "#emissions", '#AIRS',
         '#airpollutionawareness', "#natureishealing", '#airpollutioncontrol', '#airparticles', '#Aerosals',
         '#smog', "#blueskychallenge", "#emissions", "#hvac", "#WorldCleanAirDay", '#cleanairforall',
         '#cleanair4all']

hashtegi = []
newhash = set()

for value in df.hashtags:
    hastaglist = value.split(',')
    for tag in hastaglist:
        hasht = '#' + tag
        tagh = hasht.replace(' ', '')
        hashtegi.append(tagh)

for hashnew in hashtegi:
    if hashnew not in hash3:
        newhash.add(hashnew)

print(newhash)

newlist = ['#airpollution', '#ClimateAction', '#AfricaAQ', '#cleanAirforall', '#climateaction', '#endcoal',
           '#NatureIsHealing', '#CleanAirJoburg', '#blueskies', '#CleanTheAir', '#FreshAir', '#SolutionToPollution',
           '#Airpollution', '#StareDownOnPollution', '#CleanAirTshwane', '#airqualitymeasurement', '#CO2emissions',
           '#CleanAirJ', '#endCoal', '#cleanairjoburg', '#KlaCleanAir', '#CleanAirForAll', '#FreshAIR',
           '#CleanAirForBlueSkies', '#cleanairday', '#beatairpollution', '#worldcleanairday']
