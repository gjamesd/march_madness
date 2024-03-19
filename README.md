# READ ME 

This is a ui designed to help fill out the NCAA March Madness bracket, the goal is to run the model in the background and the user can simply fill out their bracket. 

# Model Explanation
The model used in this project was an XGBoosted Classification model. 
    This choice was made because it is a supervised approach. Intrinsic to 
    predicting which team wins a game is understanding why they won. As a 
    supervised learning algorithm, XGBoost provides us with the importance 
    of each feature, helping us understand why a game was classified as an upset or not.
    
    The data used in this model comes from Kaggle, which provides detailed 
    information on each regular season and postseason game since 2003. The 
    specific dataset used in this architecture includes regular season averages 
    for statistics like 3-pointers made and missed, turnovers, points scored, 
    number of regular season wins and losses, conference affiliation, average 
    points given up, etc.
    
    The model takes the regular season data for two teams playing against each 
    other and then looks at past tournaments to predict, based on historical 
    NCAA Tournament games, whether the higher-seeded team will win or lose when 
    considering the regular season data.
    
    The higher seed, in this case, refers to the rank order of teams 1-64 who 
    make the field of 64. Therefore, the user must select the play-in games to 
    run the model. If a team is not in the field of 64, it does not receive a 
    "power seed." This prevents scenarios like a 1 vs. 1 seed matchup in the 
    Final Four or a 2 vs. 2 matchup, etc. Each tournament team is assigned a 
    power seed from 1-64. The outcome variable for this project is whether the 
    higher "power seed" wins.

