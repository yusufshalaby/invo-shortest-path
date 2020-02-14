import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_paths(ref_path_file,pat_path_file):
    
    #1. Define steps
    

    steps = ['EXTRA CONSULT', 'ED VISIT', 'ENDOSCOPY', 'ABDOMEN CT', 'ABDOMEN MRI/US', 'PELVIS CT','PELVIS MRI/US', 'CHEST IMAGING', 'RESECTIVE']

    end_steps = {'RESECTIVE END':steps,'MO CONSULT END':steps,'CHEMO PARTIAL':steps,'CHEMO COMPLETE':steps}
    
    #2. Get reference paths
    
    path_file = open(ref_path_file, "r")

    path_text = path_file.read()
    path_lines = path_text.splitlines()

    path_lines = list(map(lambda line: line.strip('""'), path_lines))

    rawpaths = list(map(lambda path: path.split("-"), path_lines))
        
    pathsdf = pd.DataFrame([(i,rawpaths[i][j],j) for i in range(len(rawpaths)) 
                        for j in range(len(rawpaths[i]))], columns = ['ID','Event','Order'])
    
    stepsdict = {}
    pathsdf = pathsdf[~pathsdf.Event.isin(['GI CONSULT','SURGERY CONSULT','MO CONSULT','CHEMO TREATMENT'])]
    stepsdict['FULL ENDOSCOPY'] = 'ENDOSCOPY'
    stepsdict['PART ENDOSCOPY'] = 'ENDOSCOPY'
    stepsdict['ABDOMINAL MRI'] = 'ABDOMEN MRI/US'
    stepsdict['ABDOMEN ULTRASOUND'] = 'ABDOMEN MRI/US'
    stepsdict['CHEST CT'] = 'CHEST IMAGING'
    stepsdict['CHEST X RAY'] = 'CHEST IMAGING'

    pathsdf['Step'] = pathsdf['Event'].apply(lambda x: stepsdict[x] if x in stepsdict.keys() else x)

    pathids= list(set(pathsdf.ID))
    pathtuples = [tuple(pathsdf[pathsdf.ID==pathid].Step) for pathid in pathids]
    reference_paths = [['START'] + list(p) + ['CHEMO COMPLETE','END'] for p in set(pathtuples)]
    reference_paths = [path for path in reference_paths if path.count('CHEST IMAGING')==1]
    
    
    #3. Get patient paths
    
    #import data
    df = pd.read_csv(pat_path_file)

    #filter for stage III patients
    df3 = df.loc[df.stage_long.isin(['IIIA','IIIB','IIIC'])]
    
    # indices of patients that survived or died
    patients_that_died = df3.loc[df3.event2=='DEATH'].ID.unique()
    patients_that_survived = df3.loc[~df3.ID.isin(patients_that_died)].ID.unique()

    #Path refinement for stage III patients
    cut_threshold = df3.loc[(df3.days_btw_eventdiag>=-30) & (df3.days_btw_eventdiag<=365)].copy()
   
    df3_chemo = cut_threshold.loc[(cut_threshold.event2=='CHEMO TREATMENT')].copy()
    df3_chemo = df3_chemo.merge(pd.DataFrame(df3_chemo.groupby('ID').cumcount()+1,columns=['n_chemo']),left_index=True,
                                        right_index=True,how='outer')

    cut_threshold = cut_threshold.merge(
        pd.DataFrame(df3_chemo.loc[df3_chemo.n_chemo==6].groupby('ID').days_btw_eventdiag.min()).rename(
            {'days_btw_eventdiag':'sixth_chemo'},axis=1),left_on='ID',right_index=True,how='outer')

    cut_threshold.loc[:,'sixth_chemo'] = cut_threshold.sixth_chemo.fillna(1000).copy()
    cut_threshold = cut_threshold.loc[cut_threshold.days_btw_eventdiag<=cut_threshold.sixth_chemo]
    
    cut_threshold = cut_threshold.merge(
        pd.DataFrame(cut_threshold.loc[cut_threshold.event2=='GI CONSULT'].groupby('ID').cumcount()+1,columns=['n_gi_consult']),
        left_index=True,right_index=True,how='outer')
    cut_threshold = cut_threshold.merge(
        pd.DataFrame(cut_threshold.loc[cut_threshold.event2=='SURGERY CONSULT'].groupby('ID').cumcount()+1,columns=['n_s_consult']),
        left_index=True,right_index=True,how='outer')
    cut_threshold = cut_threshold.merge(
        pd.DataFrame(cut_threshold.loc[cut_threshold.event2=='MO CONSULT'].groupby('ID').cumcount()+1,columns=['n_mo_consult']),
        left_index=True,right_index=True,how='outer')
    
    cut_threshold.loc[:,['n_gi_consult','n_s_consult','n_mo_consult']] = cut_threshold[['n_gi_consult','n_s_consult','n_mo_consult']].fillna(0)
    
    cut_threshold.loc[:,'event2'] = np.where((cut_threshold.n_gi_consult > 1) | (cut_threshold.n_s_consult > 1) | (cut_threshold.n_mo_consult > 1),
                               'EXTRA CONSULT',cut_threshold.event2)
    
    cut_threshold = cut_threshold.merge(cut_threshold.loc[cut_threshold.event2=='PALLIATIVE CUTOFF'][['ID',
            'days_btw_eventdiag']].rename({'days_btw_eventdiag':'palliative_cutoff'},axis=1),
         how='outer')
    cut_threshold.loc[:,'palliative_cutoff'] = cut_threshold.palliative_cutoff.fillna(df.days_btw_eventdiag.max()+1)
    cut_threshold = cut_threshold.loc[cut_threshold.days_btw_eventdiag < cut_threshold.palliative_cutoff]
    
    patients_that_had_resective = cut_threshold.loc[cut_threshold.event2=='RESECTIVE'].ID.unique()
    
    df_refined = cut_threshold.loc[~cut_threshold.ID.isin(set(cut_threshold[(cut_threshold.event2=='RO TREATMENT') | 
        (cut_threshold.event2=='NONRESECTIVE')].ID)) & cut_threshold.ID.isin(patients_that_had_resective)].copy()
    
    chemo_num_treatments = df_refined[df_refined.event2=='CHEMO TREATMENT'].groupby('ID').size()
    patients_completed_chemo = set(chemo_num_treatments[chemo_num_treatments>=6].index)
    patients_partial_chemo = set(chemo_num_treatments[chemo_num_treatments<6].index)
    patients_had_mo = set(df_refined.loc[df_refined.event2=='MO CONSULT'].groupby('ID').size().index).difference(
            patients_completed_chemo.union(patients_partial_chemo))
    patients_no_mo = set(df_refined.ID).difference(patients_completed_chemo.union(patients_partial_chemo).union(patients_had_mo))
        

    #ordering same day activities and removing irrelevant activities
    misc = ['ED VISIT']
    consult = ['EXTRA CONSULT']
    endoscopy = ['FULL ENDOSCOPY','PART ENDOSCOPY']
    abdo_imaging = ['ABDOMINAL MRI', 'ABDOMEN ULTRASOUND', 'ABDOMEN CT']
    pelv_imaging = ['PELVIS CT', 'PELVIS MRI', 'PELVIS ULTRASOUND']
    chest_imaging = ['CHEST CT','CHEST X RAY']
    surgery = ['RESECTIVE']
    ignore = ['GI CONSULT','SURGERY CONSULT','MO CONSULT','RO CONSULT', 'GI VISIT','SURGERY VISIT',
              'MO VISIT','BARIUM ENEMA','FOBT','RO VISIT','CT COLONOGRAPHY','SMALL BOWEL X RAY','CHEMO TREATMENT', 
              'RO TREATMENT','NONRESECTIVE','DEATH','','CHEMO COMPLETION']

    df_refined.event2 = df_refined.event2.fillna('')
    df_refined = df_refined.loc[~df_refined.event2.isin(ignore)]
    ordered_steps = [misc+consult+endoscopy+abdo_imaging+pelv_imaging+chest_imaging
                                +surgery][0]
    df_refined.loc[:,'Order'] = df_refined['event2'].apply(lambda x: ordered_steps.index(x))
    df_refined = df_refined.sort_values(['ID','days_btw_eventdiag','Order'])

    #merging activities
    stepsdict = {}
    stepsdict['FULL ENDOSCOPY'] = 'ENDOSCOPY'
    stepsdict['PART ENDOSCOPY'] = 'ENDOSCOPY'
    stepsdict['ABDOMINAL MRI'] = 'ABDOMEN MRI/US'
    stepsdict['ABDOMEN ULTRASOUND'] = 'ABDOMEN MRI/US'
    stepsdict['CHEST CT'] = 'CHEST IMAGING'
    stepsdict['CHEST X RAY'] = 'CHEST IMAGING'
    stepsdict['PELVIS MRI'] = 'PELVIS MRI/US'
    stepsdict['PELVIS ULTRASOUND'] = 'PELVIS MRI/US'     

    df_refined.loc[:,'Step'] = df_refined['event2'].apply(lambda x: stepsdict[x] if x in stepsdict.keys() else x)

    patients = set(df_refined.ID)
    patients_survived = {}
    patients_died = {}
    for patient in patients:
        if patient in patients_completed_chemo:
            if patient in patients_that_survived:
                patients_survived[patient] = ['START'] + list(df_refined[df_refined.ID==patient].Step) + ['CHEMO COMPLETE', 'END']
            else:
                patients_died[patient] = ['START'] + list(df_refined[df_refined.ID==patient].Step) + ['CHEMO COMPLETE', 'END']
        elif patient in patients_partial_chemo:
            if patient in patients_that_survived:
                patients_survived[patient] = ['START'] + list(df_refined[df_refined.ID==patient].Step) + ['CHEMO PARTIAL', 'END']
            else:
                patients_died[patient] = ['START'] + list(df_refined[df_refined.ID==patient].Step) + ['CHEMO PARTIAL', 'END']
        elif patient in patients_had_mo:
            if patient in patients_that_survived:
                patients_survived[patient] = ['START'] + list(df_refined[df_refined.ID==patient].Step) + ['MO CONSULT END', 'END']
            else:
                patients_died[patient] = ['START'] + list(df_refined[df_refined.ID==patient].Step) + ['MO CONSULT END', 'END']
        else:
            if patient in patients_that_survived:
                patients_survived[patient] = ['START'] + list(df_refined[df_refined.ID==patient].Step) + ['RESECTIVE END', 'END']
            else:
                patients_died[patient] = ['START'] + list(df_refined[df_refined.ID==patient].Step) + ['RESECTIVE END', 'END']
       
    return steps, end_steps, reference_paths, patients_survived, patients_died

def get_constraints():
    step_ranks = ['RESECTIVE','ENDOSCOPY','ABDOMEN CT','CHEST IMAGING','PELVIS CT','ABDOMEN MRI/US','EXTRA CONSULT','PELVIS MRI/US','ED VISIT']

    subpath_constraints = [
                            [['ENDOSCOPY','ABDOMEN CT','PELVIS CT','CHEST IMAGING'],
                             ['ENDOSCOPY','ABDOMEN MRI/US', 'CHEST IMAGING']],                    
                            [['RESECTIVE','CHEMO COMPLETE','END'],
                             ['RESECTIVE','CHEMO PARTIAL','END']],
                            [['RESECTIVE','CHEMO PARTIAL','END'],
                             ['RESECTIVE','MO CONSULT END','END']],
                            [['RESECTIVE','MO CONSULT END','END'],
                             ['RESECTIVE','RESECTIVE END','END']]
                            ]

    penalty_constraints = [[['RESECTIVE END','END'],1],
                           [['START','ED VISIT'],1/8],
                           [['START','ENDOSCOPY'],-1],
                           [['START','ABDOMEN CT'],1/8],
                           [['START','ABDOMEN MRI/US'],1/8],
                           [['START','PELVIS CT'],1/8],
                           [['START','PELVIS MRI/US'],1/8],
                           [['START','CHEST IMAGING'],1/8],
                           [['START','RESECTIVE'],1/8],
                           [['START','EXTRA CONSULT'],1/8]
                          ]
        
    return step_ranks, subpath_constraints, penalty_constraints

        
                