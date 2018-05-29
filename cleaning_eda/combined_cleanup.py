
# coding: utf-8

# In[ ]:


#############
### SETUP ###
#############
import pandas as pd
import numpy as np
from scipy import stats

load = pd.read_csv("../../data/train.csv")
ames = load.copy()

#################
### VAR 01-20 ###
#################

# Phoebe
# Drop columns not use
ames.drop(columns=['MSSubClass','Street','Utilities','Condition2'], inplace=True)

# MSZoning, Group-RL/RMH/FC
ames['MSZoning'] = [x if x=="RL" else "RMH" if x in ['RM','RH'] else "FC" for x in ames.MSZoning]

# LotFrontage-impute with mean
ames.LotFrontage.fillna(ames.LotFrontage.mean(),inplace=True)

# LotArea-impute with mean
ames.LotArea.fillna(ames.LotFrontage.median(),inplace=True)

# Alley, Group-Alley/NoAccess
ames.Alley.fillna("NoAccess",inplace=True)
ames['Alley']=[x if x=='NoAccess' else 'Alley' for x in ames.Alley]

# LotShape, Group-Regular/Irregular
ames['LotShape']=['Regular' if x =='Reg' else 'Irregular' for x in ames.LotShape]

# LandContour, Group-Flat/Unflat
ames['LandContour']=['Flat' if x=='Lvl' else 'Unflat' for x in ames.LandContour]

# LotConfig, Group-Inside/CulDSac/Corner
ames['LotConfig'] = [x if x in ['Inside','CulDSac'] else 'Corner' for x in ames.LotConfig]

# LandSlope, Group-Gtl/NGtl
ames['LandSlope'] = [x if x=='Gtl' else 'NGtl' for x in ames.LandSlope]

# Neighborhood, Group-SAmes/NAmes/EAmes
NAmes = ['NridgHt','NoRidge','Somerst','NWAmes','Blmngtn','Gilbert','StoneBr']
EAmes = ['BrkSide','Edwards','OldTown','IDOTRR','Sawyer','NAmes','BrDale','NPkVill']
SAmes = ['Veenker','Crawfor','ClearCr','CollgCr','SawyerW','Blueste','Timber','Mitchel','SWISU','MeadowV']
ames['Neighborhood']=['NAmes' if n in NAmes else 'EAmes' if n in EAmes else 'SAmes' for n in ames.Neighborhood]

# Condition1, Group-Norm and Pos v.s. Neg
ames['Condition1'] = ['NormP' if x in ['PosN','PosA','Norm'] else 'Neg' for x in ames.Condition1]

# BldgType, Group-OneFamily/Duplex/Townhouse
ames['BldgType'] = ['OneFamily' if x=='1Fam' else 'Duplex' if x in ['2fmCon','Duplex'] else 'Townhouse' for x in ames.BldgType]

# HouseStyle, Group-OneStory/OneStoryUp
ames['HouseStyle'] = ['OneStory' if x in ['1Story','1.5Fin','1.5Unf'] else 'OneStoryUp' for x in ames.HouseStyle]

# OverallQual, Group-LowQ/AvgQ/HighQ
ames['OverallQual'] = ['LowQ' if x<=4 else 'AvgQ' if x in [5,6,7] else 'HighQ' for x in ames.OverallQual]

# OverallCond, Group-BadC/AvgC/GoodC
ames['OverallCond'] = ['BadC' if x<=4 else 'AvgC' if x in [5,6,7] else 'GoodC' for x in ames.OverallCond]

# YearBuilt
ames['YearBuilt'] =2010 - ames['YearBuilt']

# YearRemodAdd
ames['remod']=ames['YearRemodAdd']-ames['YearBuilt']
ames['YearRemodAddGroup']=['NoRemod' if x==0 else 'Remod' for x in ames.remod]
ames.drop(columns=['remod'])

#################
### VAR 21-40 ###
#################

"""
Mike's Ames Housing Data Transformation block
Format:
# Variable: variable in ames df
# Problems: described issues (missingness, transformations, unbalanced categories)
# Solutions: transformations performed
my code
# (optional) Alternative: if the approach is low confidence, provide another strategy 
# (optional) alternative code
"""

### Variable: RoofStyle
# Problems: 6 categories, 4 of which were less than 1% of total counts
# Solution: Condense categories to most common (gable roof, 78%) and less common(Hip, Flat, etc...)
ames.RoofStyle = ['gable' if rs == 'Gable' else 'not_gable' for rs in ames.RoofStyle]
# gable is 'standard' construction type (vs hip)
# Alternative: evaluate as 'hip'; 2nd most common and a 'premium' roof style
#ames.RoofStyle = ['hip' if rs == 'Hip' else 'not_hip' for rs in ames.RoofStyle]

### Variable: RoofMatl
# Problems: 98% of roof materials are Composite Shingles
# Solution: drop, below our threshold of at least a 95%/5% class balance
ames.drop(columns = 'RoofMatl', inplace = True)

### NEW ### Variable: IsVinyl
# Problems: duplicate info between exterior1st, 2nd below
# Solution: Vinyl siding is a common, standard material for houses
vinylSd = pd.Series([1 if sdg == 'VinylSd' else 0 for sdg in ames.Exterior1st])
vinylSd2 = pd.Series([1 if sdg == 'VinylSd' else 0 for sdg in ames.Exterior2nd])
combVinyl = vinylSd + vinylSd2
isVinyl = ['yes' if val > 0 else 'no' for val in combVinyl]
ames['IsVinyl'] = isVinyl

### Variable: Exterior1st
# Problems: 
# Solution: combined above, drop the original column
ames.drop(columns = ['Exterior1st'], inplace = True)

### Variable: Exterior2nd
# Problems: 
# Solution: combined above, drop the original column
ames.drop(columns = ['Exterior2nd'], inplace = True)

### Variable: MasVnrType
# Problems: 8 missing values; 4 categories with 1 less than 1%
# Solution: Assume that it was skipped as type/area == 0; group into masvnr or no_masvnr
ames.MasVnrType.fillna('None', inplace = True)
ames.MasVnrType = ['no_masvnr' if mvt == 'None' else 'masvnr' for mvt in ames.MasVnrType]
# modify to divide into brick and stone and other
# Alternative: separate into brick/stone/none

### Variable: MasVnrArea
# Problems: 8 missing values; 59% zeros strongly skew linear model
# Solution: Assume that it was skipped as type/area == 0; MasVnrType may capture the variance
ames.drop(columns = ['MasVnrArea'], inplace = True)
# Alternative: per ecology models, map for binary (as above for type) and then take log value of the number if > 0 

### Variable: ExterQual
# Problems: 3 categories (Excellent/Fair/Poor) < 5%; note that poor is not in training set
# Solution: Combine on a 'quality' split (Excellent/Good) and (Average/Fair/Poor) [37/63% split, respectively]
ames.ExterQual = ['high' if eq in ['Ex', 'Gd'] else 'low' for eq in ames.ExterQual]

### Variable: ExterCond
# Problems: as above, unbalanced classes with 3 categories < 5%
# Solution: Combine on same 'quality' split as above. Note Average is 88%, Good is 10%.
ames.ExterCond = ['high' if ec in ['Ex', 'Gd'] else 'low' for ec in ames.ExterCond]
### Alternative: pvalue=0.0488; given 80 repeated tests this is likely a FP 
# ames.drop(columns = ['ExterCond'])

### Variable: Foundation
# Problems: unbalanced classes with 3 categories < 5%
# Solution: Given adhoc knowledge (https://goo.gl/gDjDyc), Pconc is preferred foundation so group by pconc vs other
ames.Foundation = ['pconc' if fou == 'PConc' else 'other' for fou in ames.Foundation]

### Variable: BsmtQual ## NB ## 'quality' refers to the *height* of the basement. 
# Problems: NaN is defined as 'no basement'; 2 categories < 5%
# Solution: As above, collapse Ex/Gd as 'high'. Note, not having a basement (NB) is grouped with Average(TA)/Fa/Po
ames.BsmtQual = ['high' if bq in ['Ex', 'Gd'] else 'low' for bq in ames.BsmtQual]

### Variable: BsmtCond
# Problems: NaN is defined as 'no basement'; 4 categories below
# Solution: replace NaN with 'NB'; collapse categories as above
ames.BsmtCond = ['high' if bcond in ['Ex', 'Gd'] else 'low' for bcond in ames.BsmtCond]

### Variable: BsmtExposure
# Problems: 38 missing values; opportunity to condense into binary (decent basement exposure vs poor/no exposure)
# Solution: 37/38 values had TotalBsmtSF == 0 so set to no basement; collapsed categories
ames.BsmtExposure = ['exposure' if bex in ['Av', 'Gd', 'Mn'] else 'no_exposure' for bex in ames.BsmtExposure]

### Variable: BsmtFinType1
# Problems: diverse categories; opportunity to condense into developed or undeveloped(no basement, low quality, unfinished)
# Solution: condense into developed or undeveloped basement type
ames.BsmtFinType1 = ['undeveloped' if bt in ['Unf', 'LwQ', 'NB'] else 'developed' for bt in ames.BsmtFinType1]

### Variable: BsmtFinSF1
# Problems: BsmtFinSF1 + BsmtFinSF2 + BsmtUnf = TotalBsmtSF
# Solution: drop
ames.drop(columns = ['BsmtFinSF1'], inplace = True)

### double check that BsmtFinSF1 is largest component 

### Variable: BsmtFinType2
# Problems: pvalue=0.073; not statistically significant between developed or undeveloped on SalePrice
# Solution: drop
ames.drop(columns = ['BsmtFinType2'], inplace = True)

### Variable: BsmtFinSF2
# Problems: 59% zeroes. BsmtFinSF1 + BsmtFinSF2 + BsmtUnf = TotalBsmtSF
# Solution: drop
ames.drop(columns = ['BsmtFinSF2'], inplace = True)

### Variable: BsmtUnfSF
# Problems: BsmtFinSF1 + BsmtFinSF2 + BsmtUnf = TotalBsmtSF
# Solution: drop
ames.drop(columns = ['BsmtUnfSF'], inplace = True)

### Variable: TotalBsmtSF
# Problems: small bimodal hump from zero basements, ought to be corrected
# Solution: define bsmt sizes in quartiles, using quartile splits from training set
ames.TotalBsmtSF = ['large' if sf > 1298.25 else 'mhigh' if sf > 991.50 else 'mlow' if sf > 795.75 else 'small' for sf in ames.TotalBsmtSF]
# Alternative: scale using Robust Scaler?

### Variable: Heating
# Problems: Almost 98% of categories are 'GasA' so highly unequal classes
# Solution: drop
ames.drop(columns = ['Heating'], inplace = True)

### Variable: HeatingQC
# Problems: 2 categories < %5, opportunity to condense Ex/Gd as before
# Solution: Combine on a 'quality' split (Excellent/Good) and (Average/Fair/Poor) [67/33% split, respectively]
ames.HeatingQC = ['high' if hqc in ['Ex', 'Gd'] else 'low' for hqc in ames.HeatingQC]

#################
### VAR 41-60 ###
#################

# John

def centralair(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'CentralAir': Central air conditioning, with categories 
       N    No
       Y    Yes
    Returns:
    'CentAir' -- if Y
    'NoCentAir' -- if N or missing
    """        
    Id = cols[0]
    CentralAir = cols[1]
    
    if pd.isnull(CentralAir):
        return 'NoCentAir'    
    else:
        if CentralAir == 'Y':
            return 'CentAir'
        else:
            return 'NoCentAir'
          
ames['CentralAir'] = ames[['Id','CentralAir']].apply(centralair,axis=1)

def electric(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'Electrical': Electrical system, with categories 
       SBrkr    Standard Circuit Breakers & Romex
       FuseA    Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF    60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP    60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix      Mixed
    Returns:
    'SBrkr' -- if 'SBrkr'
    'Other' -- if FuseA, FuseF, FuseP, Mix, or missing
    """        
    Id = cols[0]
    Electrical = cols[1]
    
    if pd.isnull(Electrical):
        return 'OtherCBrkr'
    else:
        if Electrical == 'SBrkr':
            return 'StdCBrkr'
        else:
            return 'OtherCBrkr'
        
ames['Electrical'] = ames[['Id','Electrical']].apply(electric,axis=1)

ames.drop(['1stFlrSF'], axis = 1, inplace = True)
ames.drop(['2ndFlrSF'], axis = 1, inplace = True)
ames.drop(['LowQualFinSF'], axis = 1, inplace = True)

def grlivarea(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'GrLivArea': Above grade (ground) living area square feet
    
    Returns:
    x -- if x is positive
    1464 -- if x is not positive (assumed to be missing; 1464 is the median in the training set)
    """        
    Id = cols[0]
    GrLivArea = cols[1]
    
    if pd.isnull(GrLivArea):
        return 1464 
    else:
        return GrLivArea
        
ames['GrLivArea'] = ames[['Id','GrLivArea']].apply(grlivarea,axis=1)

def totalbaths(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the new feature, 'FullBaths': with numeric values 
         1,2,3,...  
   
    Returns:
    '<=1FullBath' -- if <=1 total full baths
    '2FullBaths' -- if 2 total full baths
    '>=3FullBaths' -- if 3 or more total full baths
    """    
    
    Id = cols[0]
    BsmtFullBath = cols[1]
    FullBath = cols[2]
    
    total = BsmtFullBath + FullBath 

    if pd.isnull(total):
        return '<=1FullBath'
    else:
        if total <= 1:
            return '<=1FullBath'
        elif total == 2:
            return '2FullBaths'
        else:
            return '>=3FullBaths'
        
ames['FullBaths'] = ames[['Id','BsmtFullBath','FullBath']].apply(totalbaths,axis=1)

ames.drop(['BsmtFullBath','FullBath'], axis = 1, inplace = True)

def totalhalfbaths(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the new feature, 'HalfBaths': with numeric values 
         1,2,3,...  
   
    Returns:
    '0HalfBath' -- if 0 total half baths
    '>=1HalfBaths' -- if 1 or more total half baths
    """    
    
    Id = cols[0]
    BsmtHalfBath = cols[1]
    HalfBath = cols[2]
    
    total = BsmtHalfBath + HalfBath 

    if pd.isnull(total):
        return '0HalfBath'
    else:
        if total == 0:
            return '0HalfBath'
        else:
            return '>=1HalfBaths'
        
ames['HalfBaths'] = ames[['Id','BsmtHalfBath','HalfBath']].apply(totalhalfbaths,axis=1)

ames.drop(['BsmtHalfBath','HalfBath'], axis = 1, inplace = True)

def bedroomabvgr(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'BedroomAbvGr': with numeric values 
         0, 1, 2, ...  
   
    Returns:
    '<=2Bedr' -- if <=2 total bedrooms
    '=3Bedr' -- if 3 total bedrooms
    '>=4Bedr' -- if 4 or more total bedrooms
    """   
    
    Id = cols[0]
    BedroomAbvGr = cols[1]
    
    if pd.isnull(BedroomAbvGr):
        return '<=2Bedr'
    else:
        if BedroomAbvGr <= 2:
            return '<=2Bedr'
        elif BedroomAbvGr == 3:
            return '=3Bedr'
        else:
            return '>=4Bedr'
        
ames['BedroomAbvGr'] = ames[['Id','BedroomAbvGr']].apply(bedroomabvgr,axis=1)

def kitchenabvgr(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'KitchenAbvGr': with numeric values 
         0, 1, 2, ...  
   
    Returns:
    '<=1Ktchn' -- if <=1 total kitchens or missing
    '>=2Ktchn' -- if >=2 total kitchens
    """           
    Id = cols[0]
    KitchenAbvGr = cols[1]
    
    if pd.isnull(KitchenAbvGr):
        return '<=1Ktchn' 
    else:
        if KitchenAbvGr <= 1:
            return '<=1Ktchn'
        else:
            return '>=2Ktchn'
        
ames['KitchenAbvGr'] = ames[['Id','KitchenAbvGr']].apply(kitchenabvgr,axis=1)

def kitchenqual(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'KitchenQual': with categories 
               Ex   Excellent
               Gd   Good
               TA   Typical/Average
               Fa   Fair
               Po   Poor
    Returns:
    'AveKtchnQ' -- if average or worse kitchens or missing
    'GdKtchnQ' -- if good kitchens
    'ExKtchnQ' -- if excellent kitchens
    """               
    
    Id = cols[0]
    KitchenQual = cols[1]
    
    if pd.isnull(KitchenQual):
        return 'AveKtchnQ' 
    else:
        if KitchenQual == 'Gd':
            return 'GdKtchnQ'
        elif KitchenQual == 'Ex':
            return 'ExKtchnQ'
        else:
            return 'AveKtchnQ'
        
ames['KitchenQual'] = ames[['Id','KitchenQual']].apply(kitchenqual,axis=1)

def totrmsabvgrd(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'TotRmsAbvGrd': Total rooms 
               above grade (does not include bathrooms) 
    Returns:
    '<=4TotRms' -- if <=4 total rooms or missing
    '567TotRms' -- if 5, 6, 7 total rooms
    '>=8TotRms' -- if >=8 total rooms   
    """   
    
    Id = cols[0]
    TotRmsAbvGrd = cols[1]
    
    if pd.isnull(TotRmsAbvGrd):
        return '567TotRms'
    else:
        if TotRmsAbvGrd <= 4:
            return '<=4TotRms'
        elif TotRmsAbvGrd in [5, 6, 7]:
            return '567TotRms'
        else:
            return '>=8TotRms'    
        
ames['TotRmsAbvGrd'] = ames[['Id','TotRmsAbvGrd']].apply(totrmsabvgrd,axis=1)

def functional(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'Functional': Home functionality 
               (Assume typical unless deductions are warranted) with categories  

               Typ    Typical Functionality
               Min1   Minor Deductions 1
               Min2   Minor Deductions 2
               Mod    Moderate Deductions
               Maj1   Major Deductions 1
               Maj2   Major Deductions 2
               Sev    Severely Damaged
               Sal    Salvage only

    Returns:
    'NTypFunc' -- if not typical functionality or missing
    'TypFunc' -- if typical functionality
    """       
    
    Id = cols[0]
    Functional = cols[1]
    
    if pd.isnull(Functional):
        return 'NTypFunc'
    else:
        if Functional == 'Typ':
            return 'TypFunc'
        else:
            return 'NTypFunc'
        
ames['Functional'] = ames[['Id','Functional']].apply(functional,axis=1)

def fireplaces(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'Fireplaces': Number of fireplaces 

    Returns:
    'Fireplace' -- if has fireplace
    'NoFireplace' -- if no fireplace or missing
    """       
    
    Id = cols[0]
    Fireplace = cols[1]
    
    if pd.isnull(Fireplace):
        return 'NoFireplace'
    else:
        if Fireplace > 0:
            return 'Fireplace'
        else:
            return 'NoFireplace'
        
ames['Fireplaces'] = ames[['Id','Fireplaces']].apply(fireplaces,axis=1)

def fireplacequ(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'FireplaceQu': Fireplace quality 
    
       Ex   Excellent - Exceptional Masonry Fireplace
       Gd   Good - Masonry Fireplace in main level
       TA   Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa   Fair - Prefabricated Fireplace in basement
       Po   Poor - Ben Franklin Stove
       NA   No Fireplace
       
    Returns:
    'GdFireplace' -- if Gd or Ex
    'OthFireplace' -- if not Gd or Ex
    """       
    
    Id = cols[0]
    FireplaceQu = cols[1]
    
    if pd.isnull(FireplaceQu):
        return 'OthFireplace'
    else:
        if FireplaceQu in ['Gd','Ex']:
            return 'GdFireplace'
        else:
            return 'OthFireplace'
        
ames['FireplaceQu'] = ames[['Id','FireplaceQu']].apply(fireplacequ,axis=1)

def garagetype(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'GarageType': Garage location 
    
       2Types   More than one type of garage
       Attchd   Attached to home
       Basment  Basement Garage
       BuiltIn  Built-In (Garage part of house - typically has room above garage)
       CarPort  Car Port
       Detchd   Detached from home
       NA       No Garage
       
    Returns:
    'Attached' -- if Attched, BuiltIn, Basment, or 2Types
    'Detached' -- if Detchd, Carport, or missing
    """       
    
    Id = cols[0]
    GarageType = cols[1]
    
    if pd.isnull(GarageType):
        return 'Detached'
    else:
        if GarageType in ['Attchd', 'BuiltIn', 'Basment', '2Types']:
            return 'Attached'
        else:
            return 'Detached'
        
ames['GarageType'] = ames[['Id','GarageType']].apply(garagetype,axis=1)

ames.drop(['GarageYrBlt'], axis = 1, inplace = True)

def garagefinish(cols):
    """
    Arguments:
    cols[0] -- dataframe column value for ID
    cols[1] -- dataframe column value for the feature, 'GarageFinish': Interior finish of the garage 
    
       Fin  Finished
       RFn  Rough Finished
       Unf  Unfinished
       NA   No Garage
       
    Returns:
    'Fin' -- if Finished
    'RFn' -- if Rough Finished
    'Unf' -- if Unfinished
    'NoGarageFin'  -- if missing
    """       
    
    Id = cols[0]
    GarageFinish = cols[1]
    
    if pd.isnull(GarageFinish):
        return 'NoGarageFin'
    else:
        if GarageFinish == 'Fin':
            return 'GarageFin'
        elif GarageFinish == 'RFn':
            return 'RoughGarageFin'
        elif GarageFinish == 'Unf':
            return 'GarageUnf'
        else:
            return 'NoGarageFin'
        
ames['GarageFinish'] = ames[['Id','GarageFinish']].apply(garagefinish,axis=1)

#################
### VAR 61-80 ###
#################

# Henry

# GarageCars: Size of garage in car capacity
# 5 categories with value from 0 to 4
# combine 3 and 4 into one single category
ames['GarageCars'] = ['AtLeast3' if x>=3 else x for x in ames.GarageCars]



# GarageArea: Size of garage in square feet
# no meaningful information for this feature as it's captured with GarageCars.  Dropping it
ames.drop(columns = ['GarageArea'], inplace = True)


# GarageQual: Garage quality
#
#       Ex	Excellent
#       Gd	Good
#       TA	Typical/Average
#       Fa	Fair
#       Po	Poor
#       NA	No Garage
# 6 categories with values listed above.  Creating new feature as: good vs not_good
# Fill missing values as NG for no garage

ames.GarageQual.fillna('NG', inplace = True)
ames.GarageQual = ['GarageQual_good' if gqual in ['Ex', 'Gd','TA'] else 'GarageQual_not_good' for gqual in ames.GarageQual]



#GarageCond: Garage condition
#
#       Ex	Excellent
#       Gd	Good
#       TA	Typical/Average
#       Fa	Fair
#       Po	Poor
#       NA	No Garage
# 6 categories with values listed above.  Creating feature as: good vs not_good
# Fill missing values as NG for no garage

ames.GarageCond.fillna('NG', inplace = True)
ames.GarageCond = ['GarageCond_good' if gcond in ['Ex', 'Gd','TA'] else 'GarageCond_not_good' for gcond in ames.GarageCond]


#PavedDrive: Paved driveway
#
#       Y	Paved 
#       P	Partial Pavement
#       N	Dirt/Gravel
# 3 categories with values listed above.  Creating feature as 'Y' vs 'N'

ames.PavedDrive = ames.PavedDrive.str.replace('P', 'N')

#WoodDeckSF: Wood deck area in square feet
#Convert to Y and N to represent if there is a WoodDeck on property

ames['WoodDeckSF'] = ['WoodDeck_Yes' if x>0 else 'WoodDeck_No' for x in ames.WoodDeckSF]

#OpenPorchSF: Open porch area in square feet
#Convert to Y and N to represent if there is an OpenPorch on property

ames['OpenPorchSF'] = ['OpenPorch_Yes' if x>0 else 'OpenPorch_No' for x in ames.OpenPorchSF]


#EnclosedPorch: Enclosed porch area in square feet
#3SsnPorch: Three season porch area in square feet
#ScreenPorch: Screen porch area in square feet
#
# Need to combine the above 3 features into one.  Basically designated if the property has an enclosed porch with values 'Yes' or 'No'
# first create another feature column ('Enclosed_combined') as the sum of these 3 features
# if Enclosed_combined is greater than zero, set in the new column with values of Yes/No
#

ames['Enclosed_combined'] = (ames.EnclosedPorch + ames['3SsnPorch'] + ames.ScreenPorch)
ames['Enclosed_combined'] = ['EnclosedPorch_Yes' if x >0 else 'EnclosedPorch_No' for x in ames.Enclosed_combined]

#ames[['EnclosedPorch','3SsnPorch','ScreenPorch','Enclosed_combined']]



#PoolArea: Pool area in square feet
# over 99% of properties has no pool, dropping.
ames.drop(columns = ['PoolArea'], inplace=True)

#PoolQC: Pool quality
# over 99% of properties has no pool, dropping.
ames.drop(columns = ['PoolQC'], inplace=True)


#Fence: Fence quality
#		
#       GdPrv	Good Privacy
#       MnPrv	Minimum Privacy
#       GdWo	Good Wood
#       MnWw	Minimum Wood/Wire
#       NA	No Fence
# 5 categories with values listed above.  Creating feature as: has_fence vs no_fence

ames.Fence.fillna('NF', inplace = True)
ames.Fence = ['no_fence' if fen in ['NF' , 'NA'] else 'has_fence' for fen in ames.Fence]

#MiscFeature: Miscellaneous feature not covered in other categories
#		
#       Elev	Elevator
#       Gar2	2nd Garage (if not described in garage section)
#       Othr	Other
#       Shed	Shed (over 100 SF)
#       TenC	Tennis Court
#       NA	None
# Over 96% of the properties contain no misc feature, dropping.
ames.drop(columns = ['MiscFeature'], inplace = True)

#MiscVal: $Value of miscellaneous feature from above
# Not a meaningful feature, dropping.
ames.drop(columns = ['MiscVal'], inplace = True)

#MoSold: Month Sold (MM)
#Creating a feature for peak season vs non-peak.
#peak season months consists of May,June,July

ames.MoSold = ['peak_months' if mon in [5,6,7] else 'non_peak_months' for mon in ames.MoSold]


#YrSold: Year Sold (YYYY)
# no change to this feature

#SaleType: Type of sale
#		
#       WD 	Warranty Deed - Conventional
#       CWD	Warranty Deed - Cash
#       VWD	Warranty Deed - VA Loan
#       New	Home just constructed and sold
#       COD	Court Officer Deed/Estate
#       Con	Contract 15% Down payment regular terms
#       ConLw	Contract Low Down payment and low interest
#       ConLI	Contract Low Interest
#       ConLD	Contract Low Down
#       Oth	Other
# 10 categories values as listed above.  Will shrink to deed (WD,CWD,VWD) vs others(remaining others).

ames.SaleType = ['deed' if stype in ['WD', 'CWD','VWD'] else 'non_deed' for stype in ames.SaleType]


#SaleCondition: Condition of sale
#
#       Normal	Normal Sale
#       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
#       AdjLand	Adjoining Land Purchase
#       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
#       Family	Sale between family members
#       Partial	Home was not completed when last assessed (associated with New Homes)
# 6 categories values as listed above.  Will shrink to normal vs others.

ames.SaleCondition = ['normal' if scond in ['Normal'] else 'not_normal' for scond in ames.SaleCondition]
####end of Henry's section

#################
### SalePrice ###
#################

ames.SalePrice = np.log(ames.SalePrice)

##############
### OUTPUT ###
##############

###uncomment once finalized

#final_ames = pd.get_dummies(ames, drop_first = True)
#final_ames.to_csv("../../data/train_model.csv")

