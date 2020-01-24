from random import uniform
from csv import reader, writer, field_size_limit
from sys import maxsize

# a p value, sample this percentage of master.csv
P = 0.001

# increase the field size of CSV readers/writers
field_size_limit(maxsize)

# open master.csv as text file
with open('master.csv', 'r') as master_text:

    # create CSV reader with master.csv
    master_csv = reader(master_text, delimiter=',')

    # get the CSV header
    header = next(master_csv)

    # open psample.csv as a text file
    with open('%s.sample.csv' % str(P).split('.')[1], 'w') as psample_text:

        # create a CSV writer with psample.csv
        psample_csv = writer(psample_text, delimiter=',')

        # write the CSV header to psample.csv
        psample_csv.writerow(header)

        # loop through row arrays in master.csv
        for i, row in enumerate(master_csv):

            # print a status update
            print(i, end='\r')

            # get a float between 0 and 1
            if uniform(0, 1) < P:

                # write the row to psample.csv if less than P
                psample_csv.writerow(row)

