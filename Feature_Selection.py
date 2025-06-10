import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Loading the dataset as pandas frame
dataFrame=pd.read_csv("preprocessed_TrainData.csv")

#list of columns to be considered for the heat map
factors=['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate', 'class']

#Finding the correaltions between the features
correlations=dataFrame[factors].corr()
#Creating a figure with multiple subplots to accomdate the heat map
plt.subplots(figsize=(20, 20))

# Creating the heatmap using seaborn
sns.heatmap(correlations, square=True, vmin=-1, vmax=1, annot=False, cmap='BrBG');
#Showing the plotted heatmap
plt.show()

