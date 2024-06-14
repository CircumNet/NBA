import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, tensorflow as tf,glob
from matplotlib.widgets import TextBox

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#Load Data

''' Clean Data 
    -   Getting consistency in number of players that played
    -   Cleaning  empty and ambighious stats    
'''

'''
Target 
    -   Team and Average Age
    -   Team and Average Point Spread
    -   Team and Average 3 point attempted and made
    -   Team and Average 2 point attempted and made
    -   Team and Efficiency Average %
    -   Team and Free Throws with %
    -   Teams with Offensive and Defensive Rebounds
    -   Teams with Assists
    -   Teams with Blocks
    -   Teams with Turnovers
'''

class NBAStatsMachine:    
    def __init__(self):
        print('WELCOME TO NBA Stats Machine')
        '''
        GET TEAM AVERAGES IN EVERY RATINGS
        '''
        self.data_stream = glob.glob('.\\datasets\\player_stats\\*.csv')
        self.data = pd.DataFrame()        
        for date in self.data_stream:
            data = pd.read_csv(date)         
            self.data = pd.concat([data, self.data])
        self.team_details = {'ATL': ['Atlanta Hawks', (0.78, 0.13, 0.24, 1)], 'BOS': ['Boston Celtics', (0, 0.3, 0.19, 1)], 'BRK': ['Brooklyn Nets', (0, 0, 0, 1)], 'CHO': ['Charlotte Hornets', (0.78, 0.13, 0.24, 1)], 'CHI': ['Chicago Bulls', (0, 0.5, 0.5, 1)], 'CLE': ['Cleveland Cavaliers', (0.5, 0, 0, 1)], 'DAL': ['Dallas Mavericks', (0, 0.3, 0.63, 1)], 'DEN': ['Denver Nuggets', (0.05, 0.13, .25, 1)], 'DET': ['Detroit Pistons', (0.78, 0.06, 0.18, 1)], 'GSW': ['Golden State Warriors', (1, 0.78, 0.17, 1)], 'HOU': ['Houston Rockets', (0.78, 0.13, 0.24, 1)], 'IND': ['Indiana Pacers', (0, 0.18, 0.53, 1)], 'LAC': ['Los Angeles Clippers', (0.78, 0.13, 0.24, 1)], 'LAL': ['Los Angeles Lakers', (1, 0.84, 0, 1)], 'MEM': ['Memphis Grizzlies', (0, 0.5, 0.8, 1)], 'MIA': ['Miami Heat', (0.78, 0.13, 0.24, 1)], 'MIL': ['Milwaukee Bucks', (0.95, 0.89, 0.71, 1)], 'MIN': ['Minnesota Timberwolves', (0, 0.2, 0.4, 1)], 'NOP': ['New Orleans Pelicans', (0.7, 0.59, 0.35, 1)], 'NYK': ['New York Knicks', (0, 0.4, 0.8, 1)], 'OKC': ['Oklahoma City Thunder', (0, 0.51, 0.78, 1)], 'ORL': ['Orlando Magic', (0.77, 0.8, 0.82, 1)], 'PHI': ['Philadelphia 76ers', (0, 0.41, 0.71, 1)], 'PHO': ['Phoenix Suns', (0.9, 0.37, 0.13, 1)], 'POR': ['Portland Trail Blazers', (0.92, 0, 0.24, 1)], 'SAC': ['Sacramento Kings', (0.6, 0.19, 0.8, 1)], 'SAS': ['San Antonio Spurs', (0, 0, 0, 1)], 'TOR': ['Toronto Raptors', (0.78, 0.13, 0.24, 1)], 'UTA': ['Utah Jazz', (0, 0.18, 0.38, 1)], 'WAS': ['Washington Wizards', (0.7, 0.13, 0.13, 1)]}
    def team_stats(self,code,command):        
        if code in self.team_details.keys():
            self.team_data = self.data[self.data['Tm']==code]
            players = self.team_data['Player'].unique()
            match command:
                case 'AGE':
                    age = self.team_data.drop_duplicates('Player',keep='first')['Age']
                    print(age.max(),age.min(),round(age.mean(),2),round(age.std(),2))                    
                    return age
                case 'PTS':
                    pts = self.team_data['PTS']                    
                    stat = {}
                    for player in players:
                        stat[player]= self.team_data[self.team_data['Player']==player]['PTS'].mean()                                    
                    return stat
                case '3P':
                    total_3s = self.team_data['3P'].sum()
                    total_3a = self.team_data['3PA'].sum()
                    total_3p = round((total_3s/total_3a)*100,2)
                    stat, threep = [],{}
                    for player in players:
                        stat.append(self.team_data[self.team_data['Player']==player]['3P%'].mean())
                        threep[player] = round(self.team_data[self.team_data['Player']==player]['3P'].sum())
                        print(player,round(self.team_data[self.team_data['Player']==player]['3P%'].mean(),2))
                    x = max(stat),min(stat),sum(stat)/len(stat)
                    print(list(map(lambda u:round(u,2),x)))
                    print(total_3s,total_3a,total_3p)
                    return {'3PM':total_3s,'3PA':total_3a,'3P%':total_3p,'data':stat, 'Player': threep}
                case '2P':
                    total_2s = self.team_data['2P'].sum()
                    total_2a = self.team_data['2PA'].sum()
                    total_2p = round((total_2s/total_2a)*100,2)
                    stat, twop = [], {}
                    for player in players:
                        stat.append(self.team_data[self.team_data['Player']==player]['2P%'].mean())
                        twop[player] = round(self.team_data[self.team_data['Player']==player]['2P'].sum())
                        print(player,round(self.team_data[self.team_data['Player']==player]['2P%'].mean(),2))
                    x = max(stat),min(stat),sum(stat)/len(stat)
                    print(list(map(lambda u:round(u,2),x)))
                    print(total_2s,total_2a,total_2p)
                    return {'2PM':total_2s,'2PA':total_2a,'2P%':total_3p,'data':stat, 'Player':twop}
                case 'FG':
                    total_fgs = self.team_data['FG'].sum()
                    total_fga = self.team_data['FGA'].sum()
                    total_fgp = round((total_fgs/total_fga)*100,2)
                    stat, fg = [], {}
                    for player in players:
                        stat.append(self.team_data[self.team_data['Player']==player]['FG%'].mean())
                        fg[player] = round(self.team_data[self.team_data['Player']==player]['FG'].sum())
                        print(player,round(self.team_data[self.team_data['Player']==player]['FG%'].mean(),2))
                    x = max(stat),min(stat),sum(stat)/len(stat)
                    print(list(map(lambda u:round(u,2),x)))
                    print(total_fgs,total_fga,total_fgp)
                    return {'FGM':total_fgs,'FGA':total_fga,'FG%':total_fgp,'data':stat, 'Player': fg}
                case 'FT':
                    total_fts = self.team_data['FT'].sum()
                    total_fta = self.team_data['FTA'].sum()
                    total_ftp = round((total_fts/total_fta)*100,2)
                    stat, ft = [],{}
                    for player in players:
                        stat.append(self.team_data[self.team_data['Player']==player]['FT%'].mean())
                        ft[player] = round(self.team_data[self.team_data['Player']==player]['FT'].sum())                    
                        print(player,round(self.team_data[self.team_data['Player']==player]['FT%'].mean(),2))
                    x = max(stat),min(stat),sum(stat)/len(stat)
                    print(list(map(lambda u:round(u,2),x)))
                    print(total_fts,total_fta,total_ftp)
                    return {'FTM':total_fts,'FTA':total_fta,'FT%':total_ftp,'data':stat, 'Player': ft}
                case 'AST':
                    total_ast = self.team_data['AST'].sum()
                    stat,assist = {},{}
                    for player in players:
                        stat[player] = round(self.team_data[self.team_data['Player']==player]['AST'].mean(),2)
                        assist[player] = round(self.team_data[self.team_data['Player']==player]['AST'].sum())
                    return {'Total': total_ast, 'Data': stat,'Player': assist}
                case 'BLK':
                    total_blk = self.team_data['BLK'].sum()
                    stat,block = {},{}
                    for player in players:
                        stat[player] = self.team_data[self.team_data['Player']==player]['BLK'].mean()
                        block[player] = round(self.team_data[self.team_data['Player']==player]['BLK'].sum())
                    return {'Total': total_blk, 'Data': stat,'Player': block}
                case 'REB':
                    total_orb = self.team_data['ORB'].sum()
                    total_drb = self.team_data['DRB'].sum()
                    total_reb = total_drb + total_drb
                    orb_percent = (total_orb // total_reb) * 100
                    drb_percent = (total_drb // total_reb) * 100
                    orb_stat, drb_stat,reb = [],[],{}
                    for player in players:
                        orb_stat.append(self.team_data[self.team_data['Player']==player]['ORB'].mean())
                        drb_stat.append(self.team_data[self.team_data['Player']==player]['DRB'].mean())
                        reb[player] = [round(self.team_data[self.team_data['Player']==player]['ORB'].sum()),round(self.team_data[self.team_data['Player']==player]['DRB'].sum()),round(self.team_data[self.team_data['Player']==player]['ORB'].sum())+round(self.team_data[self.team_data['Player']==player]['DRB'].sum())]
                    return {'Total': total_reb, 'Data': reb, 'OffensiveStat': orb_stat, 'DefensiveStat': drb_stat}
                case 'TOV':
                    total_tov = self.team_data['TOV'].sum()
                    stat,tov = {},{}
                    for player in players:
                        stat[player] = round(self.team_data[self.team_data['Player']==player]['TOV'].mean(),2)
                        tov[player] = round(self.team_data[self.team_data['Player']==player]['TOV'].sum())
                    return {'Total': total_tov, 'Data': stat,'Player': tov}
                case 'EFG':
                    stat = {}
                    for player in players:
                        stat[player] = self.team_data[self.team_data['Player']==player]['eFG%'].mean()
                    return stat
        return None
    def graph(self,type):
        match type:
            case 'AGE':
                mean = {}
                for team in self.team_details.keys():
                    ans = self.team_stats(team,type)                
                    if ans is not None:                        
                        mean[team] = round(ans.mean(),2)                
                if len(mean)==len(self.team_details):
                    bar = plt.bar(mean.keys(),mean.values(),color=list(map(lambda x: x[1],self.team_details.values())))
                    plt.xticks(rotation=45)
                    plt.title(f'Average Age Across {len(self.team_details)} NBA teams')
                    plt.show()                          
            case 'PTS':                   
                while True:             
                    team = input('Enter NBA Team: ')
                    if team in self.team_details:
                        stat: dict = self.team_stats(team,type)                    
                        plt.bar(list(stat.keys()),list(map(round,stat.values())),color=self.team_details[team][1])                
                        plt.xticks(rotation=90)
                        plt.title(f'Average Point for {self.team_details[team][0]}')
                        plt.rc('axes',labelsize=6)
                        plt.tight_layout()
                        plt.show()
                    elif team == 'TOT':
                        pts = {}
                        for team in self.team_details:
                            pts[team] = round(self.data[self.data['Tm']==team]['PTS'].sum())
                        bar = plt.bar(pts.keys(),pts.values(),color=list(map(lambda x: x[1],self.team_details.values())))
                        plt.xticks(rotation=45)
                        plt.title(f'Total Points Across {len(self.team_details)} NBA teams')
                        plt.show()
            case 'AST':                    
                while True:             
                    team = input('Enter NBA Team: ')
                    if team in self.team_details:
                        stat, player = self.team_stats(team,type)['Data'], self.team_stats(team,type)['Player']
                        plt.bar(list(stat.keys()),list(map(round,stat.values())),color=self.team_details[team][1])
                        plt.xticks(rotation=90)
                        plt.rc('axes',labelsize=6)
                        plt.title(f'Average Assist Across {self.team_details[team][0]}')
                        plt.tight_layout()
                        plt.show()                        
                        plt.bar(list(player.keys()),list(map(round,player.values())),color=self.team_details[team][1])
                        plt.xticks(rotation=90)
                        plt.rc('axes',labelsize=6)
                        plt.title(f'Total Assists Across {self.team_details[team][0]}')
                        plt.tight_layout()
                        plt.show()
                    elif team == 'TOT':
                        ast = {}
                        for team in self.team_details:
                            ast[team] = round(self.data[self.data['Tm']==team]['AST'].sum())
                        bar = plt.bar(ast.keys(),ast.values(),color=list(map(lambda x: x[1],self.team_details.values())))
                        plt.xticks(rotation=45)
                        plt.title(f'Total Assists Across {len(self.team_details)} NBA teams')
                        plt.show()
            case 'BLK':                    
                while True:             
                    team = input('Enter NBA Team: ')                    
                    if team in self.team_details:
                        stat, player = self.team_stats(team,type)['Data'], self.team_stats(team,type)['Player']
                        plt.bar(list(stat.keys()),list(map(round,stat.values())),color=self.team_details[team][1])
                        plt.xticks(rotation=90)
                        plt.rc('axes',labelsize=6)
                        plt.title(f'Average Blocks Across {self.team_details[team][0]}')
                        plt.tight_layout()
                        plt.show()                        
                        plt.bar(list(player.keys()),list(map(round,player.values())),color=self.team_details[team][1])
                        plt.xticks(rotation=90)
                        plt.rc('axes',labelsize=6)
                        plt.title(f'Total Blocks Across {self.team_details[team][0]}')
                        plt.tight_layout()
                        plt.show()
                    elif team == 'TOT':
                        blk = {}
                        for team in self.team_details:
                            blk[team] = round(self.data[self.data['Tm']==team]['BLK'].sum())
                        bar = plt.bar(blk.keys(),blk.values(),color=list(map(lambda x: x[1],self.team_details.values())))
                        plt.xticks(rotation=45)
                        plt.title(f'Total Blocks Across {len(self.team_details)} NBA teams')
                        plt.show()
            case 'TOV':                    
                while True:             
                    team = input('Enter NBA Team: ')
                    if team in self.team_details:
                        stat, player = self.team_stats(team,type)['Data'], self.team_stats(team,type)['Player']
                        plt.bar(list(stat.keys()),list(map(round,stat.values())),color=self.team_details[team][1])
                        plt.xticks(rotation=90)
                        plt.rc('axes',labelsize=6)
                        plt.title(f'Average Turnovers Across {self.team_details[team][0]}')
                        plt.tight_layout()
                        plt.show()                        
                        plt.bar(list(player.keys()),list(map(round,player.values())),color=self.team_details[team][1])
                        plt.xticks(rotation=90)
                        plt.rc('axes',labelsize=6)
                        plt.title(f'Total Turnovers Across {self.team_details[team][0]}')
                        plt.tight_layout()
                        plt.show()
                    elif team == 'TOT':
                        tov = {}
                        for team in self.team_details:
                            tov[team] = round(self.data[self.data['Tm']==team]['TOV'].sum())
                        bar = plt.bar(tov.keys(),tov.values(),color=list(map(lambda x: x[1],self.team_details.values())))
                        plt.xticks(rotation=45)
                        plt.title(f'Total Turnovers Across {len(self.team_details)} NBA teams')
                        plt.show()
            case 'EFG':
                stat = {}
                while True:
                    team = input('Enter NBA Team: ')
                    if team in self.team_details:
                        stat = self.team_stats(team, type)
                        plt.bar(list(stat.keys()),list(map(round,stat.values())),color=self.team_details[team][1])
                        plt.xticks(rotation=90)
                        plt.rc('axes',labelsize=6)
                        plt.title(f'Average Efficiency Field Goal % Across {self.team_details[team][0]}')
                        plt.tight_layout()
                        plt.show()                        
                pass   
    def player_stats(self,name):

        pass    

if __name__=='__main__':
    nba = NBAStatsMachine()    
    nba.graph('EFG')