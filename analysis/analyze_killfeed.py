import pandas as pd

# Read in the killfeed.csv file
killfeed_df = pd.read_csv('killfeed.csv')

# filter text != 'no matches'
killfeed_df = killfeed_df[killfeed_df['text'] != 'no matches']

# split text column into two columns on ' killed '
killfeed_df[['killer', 'victim']] = killfeed_df['text'].str.split(' killed ', expand=True)

## add seconds column which is frame_number / 60
killfeed_df['seconds'] = killfeed_df['frame_number'] / 60

# Sort the DataFrame by killer, victim, and frame_number
killfeed_df = killfeed_df.sort_values(by=['killer', 'victim', 'frame_number'])

# Initialize a list to store distinct events
distinct_events = []

# Iterate through the DataFrame to find distinct events
for i, row in killfeed_df.iterrows():
    row['id'] = i
    if i == 0:
        distinct_events.append(row)
    else:
        if not any(
            (row['killer'] == event['killer'] and 
             row['victim'] == event['victim'] and 
             abs(row['seconds'] - event['seconds']) <= 5)
            for event in distinct_events
        ):
            distinct_events.append(row)
            

# Create a DataFrame from the list of distinct events
distinct_events_df = pd.DataFrame(distinct_events)

# Select the required columns and return the min(row) and min(frame_number) for each event
result_df = distinct_events_df.groupby(['killer', 'victim', 'id']).agg({'row': 'min', 'frame_number': 'min'}).reset_index()

# Print the result DataFrame
print(result_df.sort_values(by='frame_number').reset_index(drop=True))

## for each name, count how many times they were killer and victim
killer_counts = result_df['killer'].value_counts().reset_index()
killer_counts.columns = ['name', 'kills']
victim_counts = result_df['victim'].value_counts().reset_index()
victim_counts.columns = ['name', 'deaths']
##combine
combined_counts = pd.merge(killer_counts, victim_counts, on='name', how='outer').fillna(0)
print(combined_counts.sort_values(by = 'name').reset_index(drop=True))

