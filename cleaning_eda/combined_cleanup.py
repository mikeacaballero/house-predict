
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
ames.drop(columns=['MSSubClass','Street','Utilities','Condition2'])

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
ames['Condition1Group'] = ['NormP' if x in ['PosN','PosA','Norm'] else 'Neg' for x in ames.Condition1]

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
ames.Fence = ['no_fence' if fen in ['NF'] else 'has_fence' for fen in ames.Fence]

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

