from CareNetwork import *
from path_processing import get_paths


if __name__ == '__main__':
    
    ref_path_file = r'../../data files/pathways_in_relative_distance_III_ideal.csv'
    pat_path_file = r'../../data files/Datafile-2019-09-12/UofT transfer 01AUG19.csv'
    
    _, _, reference_paths, patients_that_survived, patients_that_died,df_refined = get_paths(ref_path_file,pat_path_file)
    
    df_refined_survived = df_refined.loc[df_refined.ID.isin(patients_that_survived)].copy()
    df_refined_died = df_refined.loc[df_refined.ID.isin(patients_that_died)].copy()
    patients_survived_paths = {patID:['START'] + list(df_refined_survived.loc[df_refined_survived.ID==patID].Step) for patID in patients_that_survived}
    patients_died_paths = {patID:['START'] + list(df_refined_died.loc[df_refined_died.ID==patID].Step) for patID in patients_that_died}
    patients_survived_statepaths = {patID:[('START', ())] + list(zip(df_refined_survived.loc[df_refined_survived.ID==patID].Step, 
                                                               df_refined_survived.loc[df_refined_survived.ID==patID].State)) for patID in patients_that_survived}
    patients_died_statepaths = {patID:[('START', ())] + list(zip(df_refined_died.loc[df_refined_died.ID==patID].Step, 
                                                               df_refined_died.loc[df_refined_died.ID==patID].State)) for patID in patients_that_died}
        
    rep_steps = {'ABDOMEN CT','ABDOMEN MRI/US', 'CHEST IMAGING', 'ED VISIT', 'ENDOSCOPY', 'EXTRA CONSULT', 'PELVIS CT', 'PELVIS MRI/US'}
    nonrep_steps = {'RESECTIVE'}
    end_steps = {'RESECTIVE END', 'MO CONSULT END', 'CHEMO PARTIAL', 'CHEMO COMPLETE'}
    trigger_steps = {'RESECTIVE'}
    step_abbr_dict = {'ABDOMEN CT':'Act','CHEST IMAGING':'C','ENDOSCOPY':'E','PELVIS CT':'P','RESECTIVE':'R','ABDOMEN MRI/US':'Amri'}
         
    #testNetwork = MultiStateModel(rep_steps, nonrep_steps, reference_paths, end_steps, trigger_steps, step_abbr_dict)   
    

    