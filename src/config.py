class Config:
    def __init__(self):
        self.length = 50
        self.questions = 10
        self.lr = 0.0005
        self.bs = 128      #128
        self.epochs = 50    #40
        self.hidden = 128
        self.layers = 1
        self.assignment = 439   # A1 = 439, A2 = 487, A3 = 492, A4 = 494, A5 - 502
        self.code_path_length = 8
        self.code_path_width = 2
        self.model_type = "DKT" #[R-Code-DKT,P-Code-DKT,DKT,Code-DKT]
        self.MAX_CODE_LEN = 100
        self.MAX_QUESTION_LEN_partI = 768  #768
        self.MAX_QUESTION_LEN_partII = 0  #128
        self.Reference_LEN = 200 # 200
        self.embedds_type = "t50"
