#Dataset rela de uma empresa "The IWSR"


import os

path = 'cities'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.json' in file:
            files.append(os.path.join(r, file))
import json

contador =  5;
cities_data = {

}
for f in files:
    print(f)
    json_file = open(f, 'r')
    data = json.load(json_file)

    city_id = data['config']['city_id'];
    city_name = data['config']['city_name'];
    total_deaths = data['original'][list(data['original'])[-1]];

    print(data['config'])
    print(total_deaths)

    cities_data[city_id] = {
        'city_name': city_name,
        'city_id': city_id,
        'total_deaths': total_deaths
    }

f = open('cities_data.json', "w+")
f.write(json.dumps(cities_data));
f.close()