import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_paths(ref_path_file,pat_path_file):
    
    #1. Define steps
    

    steps = ['EXTRA CONSULT', 'ED VISIT', 'ENDOSCOPY', 'ABDOMEN CT', 'ABDOMEN MRI/US', 'PELVIS CT','PELVIS MRI/US', 'CHEST IMAGING', 'RESECTIVE']

    end_steps = ['RESECTIVE END','MO CONSULT END','CHEMO PARTIAL','CHEMO COMPLETE']
    
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

    # filter events outside of timeline of interest
    cut_threshold = df3.loc[(df3.days_btw_eventdiag>=-30) & (df3.days_btw_eventdiag<=365)].copy()
   
    # identify the 6th chemo event for each patient that had at least 6 chemo events
    df3_chemo = cut_threshold.loc[(cut_threshold.event2=='CHEMO TREATMENT')].copy()
    df3_chemo = df3_chemo.merge(pd.DataFrame(df3_chemo.groupby('ID').cumcount()+1,columns=['n_chemo']),left_index=True,
                                        right_index=True,how='outer')

    cut_threshold = cut_threshold.merge(
        df3_chemo.loc[df3_chemo.n_chemo==6].groupby('ID',as_index=False).days_btw_eventdiag.min().rename(
            {'days_btw_eventdiag':'sixth_chemo'},axis=1),on='ID',how='outer')

    cut_threshold.loc[:,'sixth_chemo'] = cut_threshold.sixth_chemo.fillna(1000).copy()
    
    # cut all events that took place after sixth chemo
    cut_threshold = cut_threshold.loc[cut_threshold.days_btw_eventdiag<=cut_threshold.sixth_chemo]
    
    # rename all repeated consults as 'EXTRA CONSULT'
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
    
    # remove all events that took place after the 'PALLIATIVE CUTOFF' event
    cut_threshold = cut_threshold.merge(cut_threshold.loc[cut_threshold.event2=='PALLIATIVE CUTOFF'][['ID',
            'days_btw_eventdiag']].rename({'days_btw_eventdiag':'palliative_cutoff'},axis=1),
         how='outer')
    cut_threshold.loc[:,'palliative_cutoff'] = cut_threshold.palliative_cutoff.fillna(df.days_btw_eventdiag.max()+1)
    cut_threshold = cut_threshold.loc[cut_threshold.days_btw_eventdiag < cut_threshold.palliative_cutoff]
    
    # identify all patients that had surgical resection
    patients_that_had_resective = cut_threshold.loc[cut_threshold.event2=='RESECTIVE'].ID.unique()
    
    # remove patients that had alternative types of treatment
    df_refined = cut_threshold.loc[~cut_threshold.ID.isin(set(cut_threshold[(cut_threshold.event2=='RO TREATMENT') | 
        (cut_threshold.event2=='NONRESECTIVE')].ID)) & cut_threshold.ID.isin(patients_that_had_resective)].copy()
    
    # idenify patients belonging to the four types of treatment outcomes
    chemo_num_treatments = df_refined.loc[df_refined.event2=='CHEMO TREATMENT'].groupby('ID').size()
    patients_completed_chemo = set(chemo_num_treatments.loc[chemo_num_treatments>=6].index)
    patients_partial_chemo = set(chemo_num_treatments.loc[chemo_num_treatments<6].index)
    patients_had_mo = set(df_refined.loc[df_refined.event2=='MO CONSULT'].groupby('ID').size().index).difference(
            patients_completed_chemo.union(patients_partial_chemo))
    
    
    patients_no_mo = set(df_refined.ID).difference(patients_completed_chemo.union(patients_partial_chemo).union(patients_had_mo))
        
    # identify the day of the final event for each patient
    finalstepdf = df_refined.groupby('ID',as_index=False).days_btw_eventdiag.max().rename(
            {'days_btw_eventdiag':'final_step'},axis=1)
    # identify the day of the first MO consult for each patient
    firstmodf = df_refined.loc[df_refined.event2=='MO CONSULT'].groupby('ID',as_index=False).days_btw_eventdiag.min().rename(
            {'days_btw_eventdiag':'mo_cons'},axis=1)
    # identify the day of the first chemo treatment for each patient
    partchemodf = df_refined.loc[df_refined.event2=='CHEMO TREATMENT'].groupby('ID',as_index=False).days_btw_eventdiag.min().rename(
            {'days_btw_eventdiag':'part_chemo'},axis=1)
    # identify the day of the sixth chemo treatment for each patient
    compchemodf = df_refined.loc[(df_refined.ID.isin(patients_completed_chemo)) & 
                                 (df_refined.event2=='CHEMO TREATMENT')].groupby('ID',as_index=False).days_btw_eventdiag.max().rename(
            {'days_btw_eventdiag':'comp_chemo'},axis=1)
    
    df_refined = df_refined.merge(partchemodf,on='ID',how='outer').merge(compchemodf,
                    on='ID',how='outer').merge(firstmodf,on='ID',how='outer').merge(finalstepdf,on='ID',how='outer')
    
    df_refined.loc[:,['final_step','mo_cons','part_chemo','comp_chemo']] = df_refined.loc[:,['final_step','mo_cons','part_chemo','comp_chemo']].fillna(10000)
    
    # at any given event identify the corresponding outcome
    #df_refined['End step'] = 'RESECTIVE END'
    #df_refined['End step'] = np.where(df_refined.days_btw_eventdiag >= df_refined.mo_cons, 'MO CONSULT END', df_refined['End step'])
    #df_refined['End step'] = np.where(df_refined.days_btw_eventdiag >= df_refined.part_chemo, 'CHEMO PARTIAL', df_refined['End step'])
    #df_refined['End step'] = np.where(df_refined.days_btw_eventdiag >= df_refined.comp_chemo, 'CHEMO COMPLETE', df_refined['End step'])

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
    patients_that_survived = set(patients_that_survived).intersection(patients)
    patients_that_died = set(patients_that_died).intersection(patients)
    
    
    df_outcome = df_refined[['ID','stage_long','final_step','mo_cons','part_chemo','comp_chemo']].drop_duplicates()
    df_outcome['Step'] = 'RESECTIVE END'
    df_outcome['Step'] = np.where(df_outcome.ID.isin(patients_had_mo), 'MO CONSULT END',df_outcome.Step)
    df_outcome['Step'] = np.where(df_outcome.ID.isin(patients_partial_chemo), 'CHEMO PARTIAL',df_outcome.Step)
    df_outcome['Step'] = np.where(df_outcome.ID.isin(patients_completed_chemo),'CHEMO COMPLETE',df_outcome.Step)
    #df_outcome['End step'] = df_outcome.Step
    df_outcome['days_btw_eventdiag'] = df_outcome.final_step
    #df_outcome['days_btw_eventdiag'] = np.where(df_outcome.ID.isin(patients_had_mo),
    #          df_outcome.mo_cons,df_outcome.days_btw_eventdiag)
    #df_outcome['days_btw_eventdiag'] = np.where(df_outcome.ID.isin(patients_partial_chemo),
    #          df_outcome.part_chemo,df_outcome.days_btw_eventdiag)
    df_outcome['days_btw_eventdiag'] = np.where(df_outcome.ID.isin(patients_completed_chemo),
              df_outcome.comp_chemo,df_outcome.days_btw_eventdiag)
    
    df_refined = df_refined.drop(['event2','sixth_chemo','n_gi_consult','n_s_consult','n_mo_consult',
                        'palliative_cutoff','Order','final_step','mo_cons','part_chemo','comp_chemo'],axis=1)
    
    df_outcome = df_outcome.drop(['final_step','mo_cons','part_chemo','comp_chemo'],axis=1)
    
    df_refined = df_refined.append(df_outcome,ignore_index=True,sort=False).sort_values(['ID','days_btw_eventdiag']).reset_index(drop=True)
    df_refined['Prev_step'] = df_refined.groupby('ID').Step.shift().fillna('START')
    df_refined['Order'] = df_refined.groupby('ID').cumcount()
    unique_path_steps = [set(reference_paths[i]).difference(set([step for j in range(len(reference_paths)) 
                                for step in reference_paths[j] if j != i])) for i in range(len(reference_paths))]
    step_path_dict = {step:i for i in range(len(unique_path_steps)) for step in unique_path_steps[i]}
    df_ref_assignments = df_refined.loc[df_refined.Step.isin(set.union(*unique_path_steps))].groupby(['ID','Step'],
                                            as_index=False).Order.min().sort_values(['ID','Order']).groupby('ID').head(1).copy()
    df_ref_assignments['Reference_path'] = df_ref_assignments.Step.apply(lambda x: step_path_dict[x])
    pat_ref_dict = {**{patID:ref for (patID,ref) in df_ref_assignments[['ID','Reference_path']].values},**{patID:0 for patID in set(df_refined.ID).difference(set(df_ref_assignments.ID))}}
    df_refined['Reference_path'] = df_refined.ID.apply(lambda x:pat_ref_dict[x])
    
    first_step = df_refined.loc[df_refined.apply(lambda x: x['Step'] in reference_paths[pat_ref_dict[x['ID']]],axis=1)].groupby(['ID','Step'],as_index=False).Order.min()
    first_step['First_concordant_step'] = True
    
    df_ref = pd.DataFrame({'Prev_step':[step for i in range(len(reference_paths)) for step in reference_paths[i][:-1]],
                       'Step':[step for i in range(len(reference_paths)) for step in reference_paths[i][1:]],
                       'Reference_path':[k for j in [[i]*(len(reference_paths[i])-1) for i in range(len(reference_paths))] for k in j]})
    #df_ref['Concordant_order'] = df_ref.groupby('Reference_path').cumcount()
    df_ref['Concordant_edge'] = True
    
    df_refined = df_refined.merge(df_ref,on=['Prev_step','Step','Reference_path'],how='left').merge(first_step,on=['ID','Step','Order'],
                                        how='left').sort_values(['stage_long','ID','Order']).reset_index(drop=True)
    df_refined.loc[:,['Concordant_edge','First_concordant_step']] = df_refined[['Concordant_edge','First_concordant_step']].fillna(False)
    #df_refined.loc[:,'Concordant_order'] = df_refined.Concordant_order.fillna(-1)
    
    df_refined['First_concordant_prev_step'] = df_refined.groupby('ID').First_concordant_step.shift().fillna(True)
    #df_refined.loc[df_refined.Step=='END',['First_concordant_step']] = True
    
    return steps, end_steps, reference_paths, patients_that_survived, patients_that_died, df_refined
    
def stateBasedProcessing(reference_paths, df_refined):

    paths = []
    for patID in df_refined.ID.unique():
        path = []
        for index,row in df_refined.loc[df_refined.ID==patID].iterrows():
            if row['First_concordant_prev_step']:
                path.append(row['Prev_step'])
            paths.append(tuple(path))
    df_refined['Completed']=paths
    
    df_refined['State'] = df_refined.Completed.map(lambda x: tuple(sorted(x[1:])))
    #df_refined.loc[df_refined.Step=='END','State'] = 'END'
    df_refined['Detour'] = ~((df_refined.Concordant_edge) & (df_refined.First_concordant_step) & (df_refined.First_concordant_prev_step))
    df_refined['Last_concordant_step'] = df_refined.Completed.apply(lambda x: x[-1])
     
    return df_refined

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

        
                