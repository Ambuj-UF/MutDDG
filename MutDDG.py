
import numpy as np
import pickle
import os
import argparse
import shutil


class DDG():
    def __init__(self, sequence, temperature=300, pH=7.0, output="output.txt"):
        try:
            os.mkdir("out")
        except:
            shutil.rmtree("out")
            os.mkdir("out")
            
        self.sequence = sequence
        self.temperature = temperature
        self.pH = pH
        self.xgb = pickle.load(open("model_out_reg.pkl", "rb"))
        self.FEATURES()
        
        with open("data.fasta", "w") as fp:
            fp.write(">data\n")
            fp.write("%s" %self.sequence)
            
        self.get_spider()
        
        with open(output, "w") as fp:
            for i, aa in enumerate(self.sequence):
                nat = aa
                for mutaa in list(self.Grantham_dict.keys()):
                    mut = mutaa
                    if nat == mut:
                        continue
                    else:
                        out = self.predict_ddg(nat, mut, i+1)
                        fp.write("%s\t%s\n" %(nat+str(i+1)+mut, np.around(out[0], 2)))
                    
        os.remove("data.fasta")
                    
        return None
        
        
        
    def FEATURES(self):
        self.data_MatrixKF = {
                        "A": [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.2, -0.08, 0.21, -0.48],
                        "R": [0.22, 1.27, 1.37, 1.87, -1.7, 0.46, 0.92, -0.39, 0.23, 0.93],
                        "N": [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.1, -1.73],
                        "D": [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.7],
                        "C": [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.1],
                        "Q": [-0.47, 0.24, 0.07, 1.1, 1.1, 0.59, 0.84, -0.71, -0.03, -2.33],
                        "E": [-1.45, 0.19, -1.61, 1.17, -1.31, 0.4, 0.04, 0.38, -0.35, -0.12],
                        "G": [1.46, -1.96, -0.23, -0.16, 0.1, -0.11, 1.32, 2.36, -1.66, 0.46],
                        "H": [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],
                        "I": [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],
                        "L": [-1.04, 0, -0.24, -1.1, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],
                        "K": [-0.34, 0.82, -0.23, 1.7, 1.54, -1.62, 1.15, -0.08, -0.48, 0.6],
                        "M": [-1.4, 0.18, -0.42, -0.73, 2, 1.52, 0.26, 0.11, -1.27, 0.27],
                        "F": [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.1, 1.71, -0.44],
                        "P": [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.3, -2.3, 0.74, -0.28],
                        "S": [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],
                        "T": [0.26, -0.7, 1.21, 0.63, -0.1, 0.21, 0.24, -1.15, -0.56, 0.19],
                        "W": [0.3, 2.1, -0.72, -1.57, -1.16, 0.57, -0.48, -0.4, -2.3, -0.6],
                        "Y": [1.38, 1.48, 0.8, -0.56, 0, -0.68, -0.31, 1.03, -0.05, 0.53],
                        "V": [-0.74, -0.71, 2.04, -0.4, 0.5, -0.81, -1.07, 0.06, -0.46, 0.65]
                        }

        self.Grantham_dict = {
                'A': {'A': 0.0, 'R': 112.0, 'N': 111.0, 'D': 126.0, 'C': 195.0, 'Q': 91.0, 'E': 107.0, 'G': 60.0, 'H': 86.0, 'I': 94.0, 'L': 96.0, 'K': 106.0, 'M': 84.0, 'F': 113.0, 'P': 27.0, 'S': 99.0, 'T': 58.0, 'W': 148.0, 'Y': 112.0, 'V': 64.0},
                'R': {'A': 112.0, 'R': 0.0, 'N': 86.0, 'D': 96.0, 'C': 180.0, 'Q': 43.0, 'E': 54.0, 'G': 125.0, 'H': 29.0, 'I': 97.0, 'L': 102.0, 'K': 26.0, 'M': 91.0, 'F': 97.0, 'P': 103.0, 'S': 110.0, 'T': 71.0, 'W': 101.0, 'Y': 77.0, 'V': 96.0},
                'N': {'A': 111.0, 'R': 86.0, 'N': 0.0, 'D': 23.0, 'C': 139.0, 'Q': 46.0, 'E': 42.0, 'G': 80.0, 'H': 68.0, 'I': 149.0, 'L': 153.0, 'K': 94.0, 'M': 142.0, 'F': 158.0, 'P': 91.0, 'S': 46.0, 'T': 65.0, 'W': 174.0, 'Y': 143.0, 'V': 133.0},
                'D': {'A': 126.0, 'R': 96.0, 'N': 23.0, 'D': 0.0, 'C': 154.0, 'Q': 61.0, 'E': 45.0, 'G': 94.0, 'H': 81.0, 'I': 168.0, 'L': 172.0, 'K': 101.0, 'M': 160.0, 'F': 177.0, 'P': 108.0, 'S': 65.0, 'T': 85.0, 'W': 181.0, 'Y': 160.0, 'V': 152.0},
                'C': {'A': 195.0, 'R': 180.0, 'N': 139.0, 'D': 154.0, 'C': 0.0, 'Q': 154.0, 'E': 170.0, 'G': 159.0, 'H': 174.0, 'I': 198.0, 'L': 198.0, 'K': 202.0, 'M': 196.0, 'F': 205.0, 'P': 169.0, 'S': 112.0, 'T': 149.0, 'W': 215.0, 'Y': 194.0, 'V': 192.0},
                'Q': {'A': 91.0, 'R': 43.0, 'N': 46.0, 'D': 61.0, 'C': 154.0, 'Q': 0.0, 'E': 29.0, 'G': 87.0, 'H': 24.0, 'I': 109.0, 'L': 113.0, 'K': 53.0, 'M': 101.0, 'F': 116.0, 'P': 76.0, 'S': 68.0, 'T': 42.0, 'W': 130.0, 'Y': 99.0, 'V': 96.0},
                'E': {'A': 107.0, 'R': 54.0, 'N': 42.0, 'D': 45.0, 'C': 170.0, 'Q': 29.0, 'E': 0.0, 'G': 98.0, 'H': 40.0, 'I': 134.0, 'L': 138.0, 'K': 56.0, 'M': 126.0, 'F': 140.0, 'P': 93.0, 'S': 80.0, 'T': 65.0, 'W': 152.0, 'Y': 122.0, 'V': 121.0},
                'G': {'A': 60.0, 'R': 125.0, 'N': 80.0, 'D': 94.0, 'C': 159.0, 'Q': 87.0, 'E': 98.0, 'G': 0.0, 'H': 98.0, 'I': 135.0, 'L': 138.0, 'K': 127.0, 'M': 127.0, 'F': 153.0, 'P': 42.0, 'S': 56.0, 'T': 59.0, 'W': 184.0, 'Y': 147.0, 'V': 109.0},
                'H': {'A': 86.0, 'R': 29.0, 'N': 68.0, 'D': 81.0, 'C': 174.0, 'Q': 24.0, 'E': 40.0, 'G': 98.0, 'H': 0.0, 'I': 94.0, 'L': 99.0, 'K': 32.0, 'M': 87.0, 'F': 100.0, 'P': 77.0, 'S': 89.0, 'T': 47.0, 'W': 115.0, 'Y': 83.0, 'V': 84.0},
                'I': {'A': 94.0, 'R': 97.0, 'N': 149.0, 'D': 168.0, 'C': 198.0, 'Q': 109.0, 'E': 134.0, 'G': 135.0, 'H': 94.0, 'I': 0.0, 'L': 5.0, 'K': 102.0, 'M': 10.0, 'F': 21.0, 'P': 95.0, 'S': 142.0, 'T': 89.0, 'W': 61.0, 'Y': 33.0, 'V': 29.0},
                'L': {'A': 96.0, 'R': 102.0, 'N': 153.0, 'D': 172.0, 'C': 198.0, 'Q': 113.0, 'E': 138.0, 'G': 138.0, 'H': 99.0, 'I': 5.0, 'L': 0.0, 'K': 107.0, 'M': 15.0, 'F': 22.0, 'P': 98.0, 'S': 145.0, 'T': 92.0, 'W': 61.0, 'Y': 36.0, 'V': 32.0},
                'K': {'A': 106.0, 'R': 26.0, 'N': 94.0, 'D': 101.0, 'C': 202.0, 'Q': 53.0, 'E': 56.0, 'G': 127.0, 'H': 32.0, 'I': 102.0, 'L': 107.0, 'K': 0.0, 'M': 95.0, 'F': 102.0, 'P': 103.0, 'S': 121.0, 'T': 78.0, 'W': 110.0, 'Y': 85.0, 'V': 97.0},
                'M': {'A': 84.0, 'R': 91.0, 'N': 142.0, 'D': 160.0, 'C': 196.0, 'Q': 101.0, 'E': 126.0, 'G': 127.0, 'H': 87.0, 'I': 10.0, 'L': 15.0, 'K': 95.0, 'M': 0.0, 'F': 28.0, 'P': 87.0, 'S': 135.0, 'T': 81.0, 'W': 67.0, 'Y': 36.0, 'V': 21.0},
                'F': {'A': 113.0, 'R': 97.0, 'N': 158.0, 'D': 177.0, 'C': 205.0, 'Q': 116.0, 'E': 140.0, 'G': 153.0, 'H': 100.0, 'I': 21.0, 'L': 22.0, 'K': 102.0, 'M': 28.0, 'F': 0.0, 'P': 114.0, 'S': 155.0, 'T': 103.0, 'W': 40.0, 'Y': 22.0, 'V': 50.0},
                'P': {'A': 27.0, 'R': 103.0, 'N': 91.0, 'D': 108.0, 'C': 169.0, 'Q': 76.0, 'E': 93.0, 'G': 42.0, 'H': 77.0, 'I': 95.0, 'L': 98.0, 'K': 103.0, 'M': 87.0, 'F': 114.0, 'P': 0.0, 'S': 74.0, 'T': 38.0, 'W': 147.0, 'Y': 110.0, 'V': 68.0},
                'S': {'A': 99.0, 'R': 110.0, 'N': 46.0, 'D': 65.0, 'C': 112.0, 'Q': 68.0, 'E': 80.0, 'G': 56.0, 'H': 89.0, 'I': 142.0, 'L': 145.0, 'K': 121.0, 'M': 135.0, 'F': 155.0, 'P': 74.0, 'S': 0.0, 'T': 58.0, 'W': 177.0, 'Y': 144.0, 'V': 124.0},
                'T': {'A': 58.0, 'R': 71.0, 'N': 65.0, 'D': 85.0, 'C': 149.0, 'Q': 42.0, 'E': 65.0, 'G': 59.0, 'H': 47.0, 'I': 89.0, 'L': 92.0, 'K': 78.0, 'M': 81.0, 'F': 103.0, 'P': 38.0, 'S': 58.0, 'T': 0.0, 'W': 128.0, 'Y': 92.0, 'V': 69.0},
                'W': {'A': 148.0, 'R': 101.0, 'N': 174.0, 'D': 181.0, 'C': 215.0, 'Q': 130.0, 'E': 152.0, 'G': 184.0, 'H': 115.0, 'I': 61.0, 'L': 61.0, 'K': 110.0, 'M': 67.0, 'F': 40.0, 'P': 147.0, 'S': 177.0, 'T': 128.0, 'W': 0.0, 'Y': 37.0, 'V': 88.0},
                'Y': {'A': 112.0, 'R': 77.0, 'N': 143.0, 'D': 160.0, 'C': 194.0, 'Q': 99.0, 'E': 122.0, 'G': 147.0, 'H': 83.0, 'I': 33.0, 'L': 36.0, 'K': 85.0, 'M': 36.0, 'F': 22.0, 'P': 110.0, 'S': 144.0, 'T': 92.0, 'W': 37.0, 'Y': 0.0, 'V': 55.0},
                'V': {'A': 64.0, 'R': 96.0, 'N': 133.0, 'D': 152.0, 'C': 192.0, 'Q': 96.0, 'E': 121.0, 'G': 109.0, 'H': 84.0, 'I': 29.0, 'L': 32.0, 'K': 97.0, 'M': 21.0, 'F': 50.0, 'P': 68.0, 'S': 124.0, 'T': 69.0, 'W': 88.0, 'Y': 55.0, 'V': 0.0}
                }

        self.PROTSUB = {
                    "A": [4, -2, -2, -2, 1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 1, 0, -3, -3,  0, -2, -1, 0, -4],
                    "R": [-2, 6, 0, -2, -4, 1, 0, -2, 1, -3, -2, 2, -1, -2, -2, 0, -1, -2, -2, -2, -1, 0, -1, -4],
                    "N": [-2, 0, 7, 1, -3, 2, -1, 0, 1, -3, -4, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4],
                    "D": [-2, -2, 1, 9, -3, 0, 3, -1, 1, -3, -4, -1, -3, -3, -1, 1, -2, -4, -3, -3, 4, 1, -1, -4],
                    "C": [1, -4, -3, -3, 14, -2, -3, -3, -3, -2, -1, -3, -1, -2, -3, -1, -1, -1, -2, -1, -3, -3, -2, -4],
                    "Q": [-1, 1, 2, 0, -2, 4, 1, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, 0, -2, 0, 3, -1, -4],
                    "E": [0, 0, -1, 3, -3, 1, 4, -3, 0, -4, -3, 0, -2, -4, -1, -1, -4, -3, -2, -2, 1, 4, -1, -4],
                    "G": [0, -2, 0, -1, -3, -2, -3, 9, -3, -4, -4, -2, -3, -4, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4],
                    "H": [-1, 1, 1, 1, -3, 0, 0, -3, 9, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4],
                    "I": [-1, -3, -3, -3, -2, -3, -4, -4, -3, 4, 3, -4, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4],
                    "L": [-1, -2, -4, -4, -1, -2, -3, -4, -3, 3, 6, -3, 2, 0, -3, -3, -1, -2, -1, 3, -4, -3, -1, -4],
                    "K": [-1, 2, 0, -1, -3, 1, 0, -2, -1, -4, -3, 6, -2, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4],
                    "M": [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -2, 8, 1, -3, -1, -1, -3, 0, 1, -3, -1, -1, -4],
                    "F": [-1, -2, -3, -3, -2, -3, -4, -4, -1, 0, 0, -3, 1, 6, -4, -2, -2, 1, 3, 0, -3, -3, -1, -4],
                    "P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -3, -4, 8, -1, -1, -4, -3, -3, -2, -1, -2, -4],
                    "S": [1, 0, 1, 1, -1, 0, -1, 0, -1, -2, -3, 0, -1, -2, -1, 4, 2, -3, -1, -2, 0, 0, 0, -4],
                    "T": [0, -1, 0, -2, -1, -1, -4, -2, -2, -1, -1, -1, -1, -2, -1, 2, 7, -2, -2, 0, -1, -1, 0, -4],
                    "W": [-3, -2, -4, -4, -1, -2, -3, -2, -2, -3, -2, -3, -3, 1, -4, -3, -2, 13, 3, -3, -4, -3, -2, -4],
                    "Y": [-3, -2, -2, -3, -2, 0, -2, -3, 2, -1, -1, -2, 0, 3, -3, -1, -2, 3, 8, -1, -3, -2, -1, -4],
                    "V": [0, -2, -3, -3, -1, -2, -2, -3, -3, 3, 3, -2, 1, 0, -3, -2, 0, -3, -1, 4, -3, -2, -1, -4],
                    "B": [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4],
                    "Z": [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4],
                    "X": [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4],
                    "*": [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1]
                    }
                    
        return True
        
    def composition(self, seq):
        com = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
        for i in seq:
            com[i] = com[i] + 1
        for key,val in com.items():
            com[key] = float(com[key])/len(seq)
        
        return list(com.values())
    
    def build_df(self, data_all):
        df = pd.DataFrame(data_all, columns = ['pH','Temperature','NatK1', 'NatK2', 'NatK3', 'NatK4', 'NatK5', 'NatK6', 'NatK7', 'NatK8', 'NatK9', 'NatK10', 'mutK1', 'mutK2', 'mutK3', 'mutK4', 'mutK5', 'mutK6', 'mutK7', 'mutK8', 'mutK9', 'mutK10', 'N+1K1', 'N+1K2', 'N+1K3', 'N+1K4', 'N+1K5', 'N+1K6', 'N+1K7', 'N+1K8', 'N+1K9', 'N+1K10', 'N-1K1', 'N-1K2', 'N-1K3', 'N-1K4', 'N-1K5', 'N-1K6', 'N-1K7', 'N-1K8', 'N-1K9', 'N-1K10', 'N+2K1', 'N+2K2', 'N+2K3', 'N+2K4', 'N+2K5', 'N+2K6', 'N+2K7', 'N+2K8', 'N+2K9', 'N+2K10', 'N-2K1', 'N-2K2', 'N-2K3', 'N-2K4', 'N-2K5', 'N-2K6', 'N-2K7', 'N-2K8', 'N-2K9', 'N-2K10', 'N+3K1', 'N+3K2', 'N+3K3', 'N+3K4', 'N+3K5', 'N+3K6', 'N+3K7', 'N+3K8', 'N+3K9', 'N+3K10', 'N-3K1', 'N-3K2', 'N-3K3', 'N-3K4', 'N-3K5', 'N-3K6', 'N-3K7', 'N-3K8', 'N-3K9', 'N-3K10', 'G-score', 'seqstruc', 'partA', 'partR', 'partN', 'partD', 'partC', 'partE', 'partQ', 'partG', 'partH', 'partI', 'partL', 'partK', 'partM', 'partF', 'partP', 'partS', 'partT', 'partW','partY','partV', 'totA', 'totR', 'totN', 'totD', 'totC', 'totE', 'totQ', 'totG', 'totH', 'totI', 'totL', 'totK', 'totM', 'totF','totP','totS','totT','totW','totY','totV', 'seqlen', 'ASA', 'Phi', 'Psi', 'Theta', 'Tau', 'HSEa_up', 'HSEa_down', 'CN', 'P(3C)', 'P(3E)', 'P(3H)', 'P(8G)', 'P(8H)', 'P(8I)', 'P(8B)', 'P(8E)', 'P(8S)', 'P(8T)', 'P(8C)', 'ss3', 'ss8'])
        return df


    def get_diss(self, filename):
        os.system("python3 run_spotdis_single.py %s" %filename)
        fdata = open(filename.split(".")[0] + ".spotds", "r")
        flag = False
        store_ss = dict()
        for lines in fdata:
            if "# Threshold" in lines:
                flag = True
                continue
            if flag == True:
                lobj = lines.strip().split("\t")
                store_ss[int(lobj[0])] = float(lobj[2])
            
        return store_ss
    
    def get_spider(self):
    
        """Runs spider software"""
    
        with open("file_list", "w") as fp:
            fp.write("%s ./%s" %("data", "data.fasta"))
        
        os.system("./impute_script_np.sh")
    
        fdata = open("out/data.i1", "r")
        self.store_feat = dict()
        flag = False
        for lines in fdata:

            if flag == True:
                if lines.strip() == "":
                    flag = False
                else:
                    lobj = [x for x in lines.strip().split(" ") if x != ""]
                    self.store_feat[int(lobj[0])] = lobj[2:]
                
            if "# AA SS SS8 " in lines:
                flag = True
                continue
    
        #os.remove("out/data.i1")
                
        return self.store_feat
    
    def get_info(self, pos):

        ss3 = self.store_feat[pos][0]
        ss8 = self.store_feat[pos][1]
        rem_feat = [float(x) for x in self.store_feat[int(pos)][2:]]
        return ss3, ss8, rem_feat


    def pull_data(self, nat, mut, pos):

        ss_cat = ["C", "E", "H"]
        mut_id = nat + str(pos) + mut

        try:
            kf1 = self.data_MatrixKF[self.sequence[pos+1]]
        except:
            kf1 = [0]*10
        try:
            kf11 = self.data_MatrixKF[self.sequence[pos-1]]
        except:
            kf11 = [0]*10
            
        try:
            kf2 = self.data_MatrixKF[self.sequence[pos+2]]
        except:
            kf2 = [0]*10
        try:
            kf22 = self.data_MatrixKF[self.sequence[pos-2]]
        except:
            kf22 = [0]*10
            
        try:
            kf3 = self.data_MatrixKF[self.sequence[pos+3]]
        except:
            kf3 = [0]*10
        try:
            kf33 = self.data_MatrixKF[self.sequence[pos-3]]
        except:
            kf33 = [0]*10

        PROTSUB_KEYS = list(self.PROTSUB.keys())
        mut_index = PROTSUB_KEYS.index(mut)
        
        ss3, ss8, rem_feat = self.get_info(pos)
        spider_data = rem_feat

        ss_info = [0, 0, 0]
        ss_info[ss_cat.index(ss3)] = 1
        
        feat = {"ph":self.pH,
            "temp":self.temperature,
            "seq":self.sequence,
            "nativeKF":self.data_MatrixKF[nat],
            "mutantKF":self.data_MatrixKF[mut],
            "KF_N-1": kf11,
            "KF_N+1" :kf1,
            "KF_N+2" :kf2,
            "KF_N-2" :kf22,
            "KF_N+3" :kf3,
            "KF_N-3" :kf33,
            "G-score":self.Grantham_dict[nat][mut],
            "protsub-score":self.PROTSUB[nat][mut_index],
            "seq_com":self.composition(self.sequence),
            "seq_com_partition":self.composition(self.sequence[max((pos-1)-10, 0):min((pos+1)+10, len(self.sequence)-1)]),
            "seq_len":len(self.sequence),
            "spider":[float(x) for x in spider_data],
            "ss": ss_info
            }
    
        obj = [[feat["ph"]]+[feat["temp"]]+feat["nativeKF"]+feat["mutantKF"]+feat["KF_N+1"]+feat["KF_N-1"]+feat["KF_N+2"]+feat["KF_N-2"]+feat["KF_N+3"]+feat["KF_N-3"]+[feat["G-score"]]+[feat["protsub-score"]]+feat["seq_com_partition"]+feat["seq_com"]+[feat["seq_len"]]+feat["spider"]+feat["ss"]]

        return obj
    
    def predict_ddg(self, nat, mut, pos):
        X = self.pull_data(nat, mut, pos)
        out = self.xgb.predict(X)
        return out
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seq', type=str,
                        help='input sequence single line entry')
    parser.add_argument('--terminallog', type=str,
                        help='redirect stdout here')
    parser.add_argument('--pH', type=float, default=7.0,
                        help='pH')
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='Temperature')
    parser.add_argument('--output', type=str, default="output.txt", help='Path and filename write output to')
    args = parser.parse_args()

    if args.terminallog:
        log = open(args.terminallog, 'w')
        sys.stdout = log
        sys.stderr = log
        
    input_sequence = args.seq

    DDG(input_sequence, args.temperature, args.pH, args.output)

    if args.terminallog:
        log.close()

if __name__ == "__main__":
    main()


