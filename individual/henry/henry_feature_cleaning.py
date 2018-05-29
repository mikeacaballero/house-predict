# GarageCars: Size of garage in car capacity
# 5 categories with value from 0 to 4
# combine 3 and 4 into one single category
ames['GarageCars'] = ['AtLeast3' if x>=3 else x for x in ames.GarageCars]



# GarageArea: Size of garage in square feet
# no meaning information for this feature.  Dropping it
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

ames.drop(columns = ['EnclosedPorch'], inplace=True)
ames.drop(columns = ['3SsnPorch'], inplace=True)
ames.drop(columns = ['ScreenPorch'], inplace=True)
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