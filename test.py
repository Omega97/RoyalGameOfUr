import os
from royal_game_of_ur import RoyalGameOfUr

games_path = 'C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning\\ur_games'

files = os.listdir(games_path)


for file_name in files:
    s = file_name
    s = s.replace('.', '_')
    s = s.split('_')
    n = int(s[1])
    print(n, file_name)
