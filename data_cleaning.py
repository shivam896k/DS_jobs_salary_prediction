import pandas as pd

df = pd.read_csv('./glassdoor_jobs.csv')
df.head()

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided' in x.lower() else 1)

df = df[df['Salary Estimate'] != '-1']

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
salary = salary.apply(lambda x: x.replace('K', '').replace('$', ''))
salary = salary.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

df['min_salary'] = salary.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = salary.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary + df.max_salary)/2

df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])

df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis=1)

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)
df['age'] = df.Founded.apply(lambda x: x if x < 0 else 2020-x)

df = df.drop(['Unnamed: 0'], axis=1)
df.to_csv('salary.csv', index=False)
