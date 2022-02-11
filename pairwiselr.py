from sklearn.linear_model import Ridge as ridge
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class KeyCovariatePairwiseLR(object):
    '''
    Remove key covariate from features
    Do not train initial LR model
    '''
    def __init__(self, alpha1=0, alpha_blend=.1, cov_steps=10,\
        func_smooth_z='sigmoid', coeff_smooth_z=10):
        self.alpha_blend_ = alpha_blend 
        self.cov_steps_ = cov_steps
        self.coeff_smooth_z = coeff_smooth_z
        if func_smooth_z == 'sigmoid':
            self.func_smooth_z = self.sigmoid_1d
        elif func_smooth_z == 'gaussian':
            self.func_smooth_z = self.gaussian_1d
        else:
            print('Linear smoothing function for z')
            self.func_smooth_z = self.linear

    def fit(self, df, col_z, col_y, cov_range_z = [], include_z_in_x = False):
        """
        Given pandas [df] and the column names of key-covariate [col_z] and label [col_y]
        Train model
        """
        col_xz = set(df.columns) - set([col_y])
        col_x = sorted(col_xz - set([col_z]))
        col_xz = sorted(col_xz)
        self.col_xz_ = col_xz
        self.col_z_ = col_z
        if include_z_in_x:
            self.col_x_ = self.col_xz_
        else:
            self.col_x_ = col_x

        # Get range of all features
        self.cov_range_ = dict()
        if len(cov_range_z)==0: # if the dynamic range of key covariate is not defined
            self.cov_range_[col_z] = self.covariate_linspace(df[col_z], n_steps=self.cov_steps_)
        else:
            self.cov_range_[col_z] = cov_range_z
        for ft in col_x:
            self.cov_range_[ft] = self.covariate_linspace(df[ft], n_steps=self.cov_steps_)
        self.cov_range_[col_z+'_plot'] = self.covariate_linspace(df[col_z], n_steps=self.cov_steps_)

        # Calculate mean of all features
        ft_median = dict()
        for ft in self.col_xz_:
            ft_median[ft] = np.median(df[ft], axis=0)
        self.ft_median_ = ft_median

        # Get matrix of all possible blends between key-cov step function and feature-specific regression
        x_basis_matrix = []

        for row in df[self.col_xz_].iterrows():
            x_vec_test = row[1][self.col_x_].values # contains all features values for one sample
            z_data = row[1][self.col_z_]

            x_basis_row = self.calc_blended_basis(x_vec_test, z_data) 
            x_basis_matrix.append(x_basis_row)
            
            # ## 
            # x_basis_all = []
            # x_samplerow = row[1][self.col_x_] # contains all features values for one sample
            # for ft in col_xz:
            #     x_data = x_samplerow[ft]
            #     x_basis_1ft = self.calc_basis(x_data, z_data, ft)
            #     x_basis_all.append(x_basis_1ft)
            # ##
        blended_model = ridge(alpha=self.alpha_blend_)
        blended_model.fit(x_basis_matrix, df[col_y])
        self.blended_model_ = blended_model
        
        # Train blended (key-cov pairwise LR) model
        blended_model = ridge(alpha=self.alpha_blend_)
        blended_model.fit(x_basis_matrix, df[col_y])
        self.blended_model_ = blended_model

    def calc_blended_basis(self, x_data_vec, z_data):
        '''
        Given one sample of: [x_data_vec] containing multiple features, and [z_data] of one key-cov value
        Calculate all possible g_i(x)*x_j, combinations between key-cov function and x_j 
        '''
        # x_data_vec = np.append(x_data_vec,1) # add constant x0 to input features <-- basis function list for x_ features
        x_basis = []
        z_unique = self.cov_range_[self.col_z_]
        n_i = len(x_data_vec)
        n_j = len(z_unique)
        for i in range(n_i):
            x_data = x_data_vec[i] # one feature
            for j in range(n_j):
                z_t = z_unique[j]
                x_basis.append(self.func_smooth_z(z_data, z_t, self.coeff_smooth_z)*x_data) # f(z, z_t)*x_data
        return x_basis # [n_i, 1]

    def calc_pairwise_reg(self, x_data, z_data, v_j):
        '''
        [only used in plotting]
        Calculate the value of pairwise interaction between feature [x_data] and key-covariate [z_data]
        Given learned [v_j] linear stacking weights on feature i corresponding to [x_data]
        '''
        x_out = 1
        z_unique = self.cov_range_[self.col_z_]
        n_j = len(v_j)
        for j in range(n_j):
            z_t = z_unique[j]
            x_out+=(v_j[j]*self.func_smooth_z(z_data, z_t, self.coeff_smooth_z)*x_data)
        return x_out
        
    def predict(self, df):
        """
        Given trained blended model and new test data, predict
        """
        # Get matrix of all possible blends between key-cov step function and feature-specific regression
        x_basis_matrix = []
        for row in df[self.col_xz_].iterrows():
            x_vec_test = row[1][self.col_x_].values
            z_data = row[1][self.col_z_]
            x_basis_row = self.calc_blended_basis(x_vec_test, z_data) 
            x_basis_matrix.append(x_basis_row)
        return self.blended_model_.predict(x_basis_matrix)      

    def feature_importance(self, x_sample):
        '''
        Given a sample, return the FI of its features
        ??
        '''
        # calculate 
        n_ft = len(self.col_xz_)
        v = self.blended_model_.coef_
        ft_names = self.col_xz_
        step_size = len(self.cov_range_[self.col_z_]) # of z
        z = x_sample[self.col_z_]
        fi = np.nan*np.zeros(n_ft)
        for i_ft in range(n_ft):
            ft_name = ft_names[i_ft]
            x = x_sample[ft_name]
            v_j = v[i_ft*step_size:(i_ft+1)*step_size]
            y_ft_x = self.calc_pairwise_reg(x, z, v_j)
            # import pdb;pdb.set_trace()
            if not ft_name==self.col_z_:
                y_ft_median = self.calc_pairwise_reg(self.ft_median_[ft_name], z, v_j)
            else:
                y_ft_median = self.calc_pairwise_reg(self.ft_median_[ft_name], \
                    self.ft_median_[ft_name], v_j)
            fi[i_ft] = y_ft_x - y_ft_median
        return fi


    def plot_pairwise_interactions(self, ft_plot=[], n_plot_cols=4, scaleup=1):
        mpl.rcParams["font.size"] = 8
        mpl.rcParams["axes.titlesize"] = 12

        ft_names = self.col_x_
        n_ft = len(ft_names)
        if not ft_plot:
            ft_plot = ft_names
        i_plot = 0
        n_plot_rows = int(n_ft/n_plot_cols) + 1 
        plt.figure(figsize=[3.54*scaleup, 2.2*n_plot_rows*scaleup/n_plot_cols],dpi=1200)

        step_size = len(self.cov_range_[self.col_z_]) # of z
        v = self.blended_model_.coef_
        z_unique = self.cov_range_[self.col_z_ +'_plot']
        for i_ft in range(n_ft):
            try:
                ft_name = ft_names[i_ft]
                # print(ft_name)
            except:
                import pdb;pdb.set_trace()
            if ft_name in ft_plot:
                v_j = v[i_ft*step_size:(i_ft+1)*step_size]
                # print(f'{ft_name}, {v_j}')            
                if ft_name == self.col_z_:
                    x_unique = self.cov_range_[ft_name+'_plot']
                else:
                    x_unique = self.cov_range_[ft_name]
                heatmap = np.zeros([len(x_unique), len(z_unique)])
                i = 0
                j = 0
                for x in x_unique:
                    j = 0
                    for z in z_unique:
                        heatmap[i,j] = self.calc_pairwise_reg(x, z, v_j)
                        j += 1
                    i += 1

                if ft_name == self.col_z_:
                    print(self.col_z_)
                    plt.subplot(n_plot_rows, n_plot_cols, i_plot+1)
                    plt.plot(z_unique,heatmap.diagonal())
                    # plt.title(ft_name)
                    plt.ylabel('pH contribution')
                    plt.xlabel(self.col_z_) 
                    # plt.xlim([6.8, 7.5])
                else:
                    plt.subplot(n_plot_rows, n_plot_cols, i_plot+1)
                    plt.pcolor(z_unique,x_unique,heatmap, cmap='Greys')
                    print_feat_name = ft_name.replace(' (mean)', '').replace('d_', '$\Delta$')
                    if 'prev_' in print_feat_name:
                        print_feat_name = print_feat_name.replace('prev_', '') + '[t-1]'
                    elif 'cur_' in ft_name:
                        print_feat_name = print_feat_name.replace('cur_', '') + '[t]'
                    plt.ylabel(print_feat_name)
                    plt.colorbar()
                    plt.xlabel(self.col_z_) 
                plt.xlabel('pH[t-1]')
                i_plot+=1
            else:
                pass
        plt.tight_layout()
 
    @staticmethod
    def step_function_1d(x, x_t):
        '''
        Implement step function and calculate output given input value [x] and threshold [x_t]
        '''
        if x<x_t:
            return 0
        else:
            return 1

    @staticmethod   
    def sigmoid_1d(x, x_t, coeff):
        return 1./(1+np.exp(-coeff*(x-x_t)))

    @staticmethod   
    def gaussian_1d(x, x_t, coeff):
        return np.exp(-((x-x_t)/coeff)**2/2)/(np.sqrt(2*np.pi)*coeff)
        
    @staticmethod   
    def linear(x, x_t, coeff):
        return x

    @staticmethod
    def covariate_linspace(x_col, n_steps=10):
        ''' 
        Given a column of single feature values [x_col], return linspace of [n_steps] values from min to max of that feature
        '''
        xmin, xmax = np.min(x_col), np.max(x_col)
        x_range = xmax - xmin
        return np.linspace(xmin-.05*x_range, xmax+.05*x_range, n_steps)
    


class KeyCovariate2d(object):
    '''
    Remove key covariate from features
    Do not train initial LR model
    '''
    def __init__(self, alpha1=0, alpha_blend=.1, cov_steps=10, func_smooth_x=None, \
        func_smooth_z='sigmoid', coeff_smooth_x=None, coeff_smooth_z=10):
        self.alpha_blend_ = alpha_blend 
        self.cov_steps_ = cov_steps
        self.coeff_smooth_z = coeff_smooth_z
        self.coeff_smooth_x = coeff_smooth_x
        # self.coeff_smooth_x = coeff_smooth_x
        if func_smooth_z == 'sigmoid':
            self.func_smooth_z = self.sigmoid_1d
        elif func_smooth_z == 'gaussian':
            self.func_smooth_z = self.gaussian_1d
        else:
            print('Linear smoothing function for z')
            self.func_smooth_z = self.linear
       
        if func_smooth_x == 'sigmoid':
            self.func_smooth_x = self.sigmoid_1d
        elif func_smooth_x == 'gaussian':
            self.func_smooth_x = self.gaussian_1d
        else:
            print('Linear smoothing function for x')
            self.func_smooth_x = self.linear

    def fit(self, df, col_z, col_y, cov_range_z = [], include_z_in_x = False):
        """
        Given pandas [df] and the column names of key-covariate [col_z] and label [col_y]
        Train model
        """
        col_xz = set(df.columns) - set([col_y])
        col_x = sorted(col_xz - set([col_z]))
        col_xz = sorted(col_xz)
        self.col_xz_ = col_xz
        self.col_z_ = col_z
        if include_z_in_x:
            self.col_x_ = self.col_xz_
        else:
            self.col_x_ = col_x

        # Get range of all features
        self.cov_range_ = dict()
        if len(cov_range_z)==0: # if the dynamic range of key covariate is not defined
            self.cov_range_[col_z] = self.covariate_linspace(df[col_z], n_steps=self.cov_steps_)
        else:
            self.cov_range_[col_z] = cov_range_z
        for ft in col_x:
            self.cov_range_[ft] = self.covariate_linspace(df[ft], n_steps=self.cov_steps_)
        self.cov_range_[col_z+'_plot'] = self.covariate_linspace(df[col_z], n_steps=self.cov_steps_)

        # Get matrix of all possible blends between key-cov step function and feature-specific regression
        x_basis_matrix = []

        for row in df[self.col_xz_].iterrows():
            x_vec_test = row[1][self.col_x_].values # contains all features values for one sample
            z_data = row[1][self.col_z_]

            x_basis_row = self.calc_blended_basis(x_vec_test, z_data) 
            x_basis_matrix.append(x_basis_row)
            
            ### 
            # x_basis_all = []
            # x_samplerow = row[1][self.col_x_] # contains all features values for one sample
            # for ft in col_xz:
            #     x_data = x_samplerow[ft]
            #     x_basis_1ft = self.calc_basis(x_data, z_data, ft)
            #     x_basis_all.append(x_basis_1ft)
            ###
        # blended_model = ridge(alpha=self.alpha_blend_)
        # blended_model.fit(x_basis_matrix, df[col_y])
        # self.blended_model_ = blended_model
        
        x_basis_all = self.calc_df_basis(df) # <<<<<<<<<<<<< FIX; how to convert to 1d and back?
        # import pdb;pdb.set_trace()        
        # Train blended (key-cov pairwise LR) model
        x_basis_all = np.reshape(x_basis_all, (len(x_basis_all), self.cov_steps_**2*len(self.col_xz_)))
        blended_model = ridge(alpha=self.alpha_blend_)
        blended_model.fit(x_basis_all, df[col_y])
        self.blended_model_ = blended_model

    def calc_blended_basis(self, x_data_vec, z_data):
        '''
        Given one sample of: [x_data_vec] containing multiple features, and [z_data] of one key-cov value
        Calculate all possible g_i(x)*x_j, combinations between key-cov function and x_j 
        '''
        # x_data_vec = np.append(x_data_vec,1) # add constant x0 to input features <-- basis function list for x_ features
        x_basis = []
        z_unique = self.cov_range_[self.col_z_]
        n_i = len(x_data_vec)
        n_j = len(z_unique)
        for i in range(n_i):
            x_data = x_data_vec[i] # one feature
            for j in range(n_j):
                z_t = z_unique[j]
                x_basis.append(self.func_smooth_z(z_data, z_t, self.coeff_smooth_z)*x_data) # f(z, z_t)*x_data
        return x_basis # [n_i, 1]
    
    def calc_df_basis(self, df):
        '''
        Calculate basis for entire [df] of dimensions {N, L+1}, N = samples, L = non-z features
        Return full basis matrix of dimensions {N, L, M, M}, where M is number of step sizes for features
        '''
        x_basis_all = []
        for row in df[self.col_xz_].iterrows():
            x_basis_sample = []
            x_samplerow = row[1][self.col_x_] # contains all features values for one sample
            z_data = row[1][self.col_z_]
            for ft in self.col_xz_:
                x_data = x_samplerow[ft]
                x_basis_1ft = self.calc_basis(x_data, z_data, ft)
                x_basis_sample.append(x_basis_1ft)
            x_basis_all.append(x_basis_sample)
        return x_basis_all

    def calc_basis(self, x_data, z_data, x_ft_name):
        '''
        Given a point feature value [x_data], its [x_ft_name], and a [z_data] key-cov value
        Calculate the matrix of basis functions
        '''
        x_unique = self.cov_range_[x_ft_name]
        z_unique = self.cov_range_[self.col_z_]
        n_steps = self.cov_steps_
        basis_mat = np.zeros((n_steps, n_steps))
        i = 0 # ticker for ordinary feature x, rows 
        for x in x_unique:
            j = 0 # ticker for z, columns
            for z in z_unique:
                basis_mat[i, j] = self.func_smooth_x(x_data, x, self.coeff_smooth_x)*self.func_smooth_z(z_data, z, self.coeff_smooth_z)
                # print(f'{i}, {j}: {basis_mat[i,j]:.4f}')
                j+=1
            i+=1
        return basis_mat


    def calc_pairwise_reg(self, x_data, z_data, v_j):
        '''
        [only used in plotting]
        Calculate the value of pairwise interaction between feature [x_data] and key-covariate [z_data]
        Given learned [v_j] linear stacking weights on feature i corresponding to [x_data]
        '''
        x_out = 1
        z_unique = self.cov_range_[self.col_z_]
        n_j = len(v_j)
        for j in range(n_j):
            z_t = z_unique[j]
            x_out+=(v_j[j]*self.func_smooth_z(z_data, z_t, self.coeff_smooth_z)*x_data)
        return x_out
        
    def predict(self, df):
        """
        Given trained blended model and new test data, predict
        """
        # Get matrix of all possible blends between key-cov step function and feature-specific regression
        # x_basis_matrix = []
        # import pdb;pdb.set_trace()
        x_basis_matrix = self.calc_df_basis(df)
        x_basis_matrix = np.reshape(x_basis_matrix, (len(x_basis_matrix), self.cov_steps_**2*len(self.col_xz_)))
        # for row in df[self.col_xz_].iterrows():
        #     x_vec_test = row[1][self.col_x_].values
        #     z_data = row[1][self.col_z_]
        #     x_basis_row = self.calc_blended_basis(x_vec_test, z_data) 
        #     # import pdb;pdb.set_trace()
        #     x_basis_matrix.append(x_basis_row)
        return self.blended_model_.predict(x_basis_matrix)      

    def plot_pairwise_interactions(self, ft_names=[], n_plot_cols=4):
        mpl.rcParams["font.size"] = 8
        mpl.rcParams["axes.titlesize"] = 12

        # n_ft = len(self.col_x_)
        # n_plot_cols = 4
        if not ft_names:
            ft_names = self.col_x_
        n_ft = len(ft_names)
        n_plot_rows = int(n_ft/n_plot_cols) + 1 
        plt.figure(figsize=[10, 2*n_plot_rows],dpi=300)
        step_size = len(self.cov_range_[self.col_z_]) # of z
        v = self.blended_model_.coef_
        z_unique = self.cov_range_[self.col_z_ +'_plot']
        for i_ft in range(n_ft):
            try:
                ft_name = ft_names[i_ft]
            except:
                import pdb;pdb.set_trace()
            v_j = v[i_ft*step_size:(i_ft+1)*step_size]
            # print(f'{ft_name}, {v_j}')            
            if ft_name == self.col_z_:
                x_unique = self.cov_range_[ft_name+'_plot']
            else:
                x_unique = self.cov_range_[ft_name]
            heatmap = np.zeros([len(x_unique), len(z_unique)])
            i = 0
            j = 0
            for x in x_unique:
                j = 0
                for z in z_unique:
                    heatmap[i,j] = self.calc_pairwise_reg(x, z, v_j)
                    j += 1
                i += 1
            
            import pdb;pdb.set_trace()

            if ft_name == self.col_z_:
                plt.subplot(n_plot_rows, n_plot_cols, i_ft+1)
                plt.plot(z_unique,heatmap.diagonal())
                plt.title(ft_name)
                plt.ylabel('pH contribution')
                plt.xlabel(self.col_z_) 
                # plt.xlim([6.8, 7.5])
            else:
                print('cmap')
                plt.subplot(n_plot_rows, n_plot_cols, i_ft+1)
                plt.pcolor(z_unique,x_unique,heatmap, cmap='Greys')
                plt.ylabel(ft_name.replace(' (mean)', ''))
                plt.colorbar()
                plt.xlabel(self.col_z_) 
                plt.title(ft_name.replace(' (mean)', ''))
            # 1*f1(z)+1*f2(z)+... not plotted
        # i_ft+=1
        # v_j = v[i_ft*step_size:(i_ft+1)*step_size]
        # f = np.zeros(20)
        # for z in z_unique:
        #     f_j = [self.func_smooth_z(_, z, self.coeff_smooth_z) for _ in z_unique]
        #     f+=f_j
        # # import pdb;pdb.set_trace()
        # plt.subplot(n_plot_rows, n_plot_cols, i_ft+1)
        # plt.plot(z_unique, f)
        # plt.title('prev_pH')
        # plt.ylabel('pH contribution')
        # plt.xlabel(self.col_z_) 
        # z_sig = [self.func_smooth_z(_, 7.2, self.coeff_smooth_z) for _ in z_unique]
        # plt.plot(z_unique, z_sig)
        # plt.title(ft_name)
        # plt.xlabel(self.col_z_) 
        plt.tight_layout()
 
    @staticmethod
    def step_function_1d(x, x_t):
        '''
        Implement step function and calculate output given input value [x] and threshold [x_t]
        '''
        if x<x_t:
            return 0
        else:
            return 1

    @staticmethod   
    def sigmoid_1d(x, x_t, coeff):
        return 1./(1+np.exp(-coeff*(x-x_t)))

    @staticmethod   
    def gaussian_1d(x, x_t, coeff):
        return np.exp(-((x-x_t)/coeff)**2/2)/(np.sqrt(2*np.pi)*coeff)
        
    @staticmethod   
    def linear(x, x_t, coeff):
        return x

    @staticmethod
    def covariate_linspace(x_col, n_steps=10):
        ''' 
        Given a column of single feature values [x_col], return linspace of [n_steps] values from min to max of that feature
        '''
        xmin, xmax = np.min(x_col), np.max(x_col)
        x_range = xmax - xmin
        return np.linspace(xmin-.05*x_range, xmax+.05*x_range, n_steps)
    
