import argparse
import functools
import logging
import re
import sys
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP
from imblearn.over_sampling import SMOTE
import numpy as np

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Naplózás (Logging) beállítása
logging.basicConfig(
    filename='data/ddos_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DDoSPipeline")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        logger.info(f"{execution_time:.4f} másodperc alatt futott le.")
        
        return result
    return wrapper

class DataLoader:
    def drop_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=[ 'Card', 'Victim IP', 'Port number', 'Attack code','Detect count', 'Significant flag', 'Packet speed', 'Data speed','Avg packet len', 'Whitelist flag'], inplace=True)

        return df

    @time_it
    def load_train(self) -> pd.DataFrame:
        logger.info(f"Tanító adatok beolvasása...")
        componentsA = pd.read_csv('data/SCLDDoS2024_SetA_components.csv')
        eventsA = pd.read_csv('data/SCLDDoS2024_SetA_events.csv')
        componentsB = pd.read_csv('data/SCLDDoS2024_SetB_components.csv')
        eventsB = pd.read_csv('data/SCLDDoS2024_SetB_events.csv')

        eventsA = self.drop_cols(eventsA)
        componentsA.drop(columns=['Card','Significant flag','Attack code', 'Time'], inplace=True)
        eventsB = self.drop_cols(eventsB)
        componentsB.drop(columns=['Card','Significant flag','Attack code', 'Time'], inplace=True)

        A = pd.merge(eventsA, componentsA, on='Attack ID')
        B = pd.merge(eventsB, componentsB, on='Attack ID')
        train = pd.concat([A, B])
        return train

    
    @time_it
    def load_test(self, set_letter: str) -> pd.DataFrame:
        logger.info(f"Adatok beolvasása...")
        components = pd.read_csv(f'data/SCLDDoS2024_Set{set_letter}_components.csv')
        events = pd.read_csv(f'data/SCLDDoS2024_Set{set_letter}_events.csv')

        events = self.drop_cols(events)
        components.drop(columns=['Card','Significant flag','Attack code', 'Time'], inplace=True)
        df = pd.merge(events, components, on='Attack ID')
        return df

class Preprocessor:
    def __init__(self):
        pass

    @time_it
    def process(self, df: pd.DataFrame, ) -> pd.DataFrame:
        logger.info("Adatok előfeldolgozása...")
        le = LabelEncoder()
        df['Type'] = le.fit_transform(df['Type'])

        df = df[df['End time'] != '0']

        df['Victim IP'] = df['Victim IP'].apply(lambda x: int(re.search(r'\d+', x).group()))
        
        columns_to_scale = ['Packet speed', 'Data speed', 'Avg packet len']

        scaler = MinMaxScaler()

        df[[col + '_normalized' for col in columns_to_scale]] = scaler.fit_transform(df[columns_to_scale])

        return df
    
    def get_time_of_day(self, hour):
        if 5 <= hour < 12:
            return 0
        elif 12 <= hour < 17:
            return 1
        elif 17 <= hour < 21:
            return 2
        else:
            return 3

    @time_it
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Új jellemzők hozzáadása...")
        
        df['Start time'] = pd.to_datetime(df['Start time'], format='mixed')
        df['End time'] = pd.to_datetime(df['End time'], format='mixed')
        df['total_seconds'] = (df['End time'] - df['Start time']).dt.total_seconds()
        df['weekday_number'] = df['Start time'].dt.weekday
        df['time_of_day'] = df['Start time'].dt.hour.apply(self.get_time_of_day)

        df['IsWeekend'] = df['weekday_number'].apply(lambda x: 1 if x >= 5 else 0)
        df['Start Hour'] = pd.to_datetime(df['Start time']).dt.hour

        # Convert hour into cyclic features
        df['Sin_Hour'] = np.sin(2 * np.pi * df['Start Hour'] / 24)
        df['Cos_Hour'] = np.cos(2 * np.pi * df['Start Hour'] / 24)

        # Convert date into cyclic features
        df['DayOfYear'] = pd.to_datetime(df['Start time']).dt.dayofyear
        df['Sin_DayOfYear'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
        df['Cos_DayOfYear'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

        # Statistical features based on 'DataSpeed' and 'DetectCount'
        df['Mean_DataSpeed'] = df.groupby('Attack ID')['Data speed'].transform('mean')
        df['Std_DataSpeed'] = df.groupby('Attack ID')['Data speed'].transform('std')
        df['Min_DataSpeed'] = df.groupby('Attack ID')['Data speed'].transform('min')
        df['Max_DataSpeed'] = df.groupby('Attack ID')['Data speed'].transform('max')

        df['Mean_PacketSpeed'] = df.groupby('Attack ID')['Packet speed'].transform('mean')
        df['Std_PacketSpeed'] = df.groupby('Attack ID')['Packet speed'].transform('std')
        df['Min_PacketSpeed'] = df.groupby('Attack ID')['Packet speed'].transform('min')
        df['Max_PacketSpeed'] = df.groupby('Attack ID')['Packet speed'].transform('max')

        df['Mean_DetectCount'] = df.groupby('Attack ID')['Detect count'].transform('mean')
        df['Std_DetectCount'] = df.groupby('Attack ID')['Detect count'].transform('std')
        df['Min_DetectCount'] = df.groupby('Attack ID')['Detect count'].transform('min')
        df['Max_DetectCount'] = df.groupby('Attack ID')['Detect count'].transform('max')

        # New features based on 'Victim IP_y', 'Port number_y', and 'Avg packet len_normalized'
        df['VictimIP_Count'] = df.groupby('Victim IP')['Victim IP'].transform('count')
        df['PortNumber_Count'] = df.groupby('Port number')['Port number'].transform('count')
        df['AvgPacketLen_Mean'] = df.groupby('Attack ID')['Avg packet len_normalized'].transform('mean')
        df['AvgPacketLen_Std'] = df.groupby('Attack ID')['Avg packet len_normalized'].transform('std')

        # New feature: DataSpeed * PacketSpeed
        df['DataSpeed_PacketSpeed'] = df['Data speed'] * df['Packet speed']

        # New feature: Port frequency
        df['PortFrequency'] = df.groupby('Port number')['Port number'].transform('count') / len(df)

        for col in ['Std_DataSpeed', 'Std_DetectCount', 'AvgPacketLen_Std']:
            indicator_col = f'{col}_Replaced'
            df[indicator_col] = df[col].isna().astype(int)  # 1 if NaN, 0 otherwise
            df[col] = df[col].fillna(0)  # Replace NaN with 0

        df["total_seconds"].replace([-88, 0], 0.1, inplace=True)
        df["packet_Total"] = df["Packet speed_normalized"] * df["Avg packet len_normalized"]

        # Create new features based on ratios
        df['PacketSpeed_Per_Second'] = df['Packet speed_normalized'] / df['total_seconds']
        df['AvgPacketLen_Per_DataSpeed'] = df['Avg packet len_normalized'] / df['Data speed_normalized']

        # Handle potential division by zero
        df['PacketSpeed_Per_Second'].replace([np.inf, -np.inf], 0, inplace=True)
        df['AvgPacketLen_Per_DataSpeed'].replace([np.inf, -np.inf], 0, inplace=True)

        # Create other ratio-based features
        df['DataSpeed_Per_TotalSeconds'] = df['Data speed_normalized'] / df['total_seconds']
        df['AvgPacketLen_Per_TotalSeconds'] = df['Avg packet len_normalized'] / df['total_seconds']

        # Handle potential division by zero for new features
        df['DataSpeed_Per_TotalSeconds'].replace([np.inf, -np.inf], 0, inplace=True)
        df['AvgPacketLen_Per_TotalSeconds'].replace([np.inf, -np.inf], 0, inplace=True)

        df['PacketSpeed_Per_Second'] = df['Packet speed_normalized'] / df['total_seconds']
        df['AvgPacketLen_Per_DataSpeed'] = df['Avg packet len_normalized'] / df['Data speed_normalized']
        df['DataSpeed_Per_TotalSeconds'] = df['Data speed_normalized'] / df['total_seconds']
        df['AvgPacketLen_Per_TotalSeconds'] = df['Avg packet len_normalized'] / df['total_seconds']

        # Handle division by zero
        for col in ['PacketSpeed_Per_Second', 'AvgPacketLen_Per_DataSpeed', 
                    'DataSpeed_Per_TotalSeconds', 'AvgPacketLen_Per_TotalSeconds']:
            df[col].replace([np.inf, -np.inf], 0, inplace=True)

        protocol_ports_extensive = {
            'Is_HTTP': 80,
            'Is_HTTPS': 443,
            'Is_FTP_Control': 21,
            'Is_FTP_Data': 20,
            'Is_SSH': 22,
            'Is_Telnet': 23,
            'Is_SMTP': 25,
            'Is_DNS': 53,
            'Is_POP3': 110,
            'Is_IMAP': 143,
            'Is_DHCP': [67, 68],
            'Is_SNMP': [161, 162],
            'Is_LDAP': 389,
            'Is_LDAPS': 636,
            'Is_SMB_CIFS': 445,
            'Is_RDP': 3389,
            'Is_SIP': [5060, 5061],
            'Is_TFTP': 69,
            'Is_MySQL': 3306,
            'Is_HTTP_Alt_8080': 8080,
            'Is_HTTP_Alt_8081': 8081,
            'Is_HTTP_Alt_80': range(8000, 8089),  # Broader range
            'Is_HTTPS_Alt_8443': 8443,
            'Is_Syslog': 514,
            'Is_IRC': 6667,
            'Is_NTP': 123,
            'Is_Kerberos': [88, 749, 750, 464],
            'Is_LDAP_Alt': 3268,
            'Is_RADIUS': [1812, 1813],
            'Is_X11': range(6000, 6064),
            'Is_SNMP_Trap': 162,
            'Is_BGP': 179,
            'Is_IMAPS_Alt': 993,
            'Is_POP3S_Alt': 995,
            'Is_Telnet_SSL': 992,
            'Is_NNTP': 119,
            'Is_NNTPS': 563,
            'Is_LDAP_TLS': 636,
            'Is_AFS': 2049,
            'Is_NFS': 2049,
            'Is_SOCKS': [1080, 1081],
            'Is_RSYNC': 873,
            'Is_CUPS': 631,
            'Is_TFTP_Alt': 69, # Already present, but for clarity
            'Is_Modbus': 502,
            'Is_CoAP': [5683, 5684],
            'Is_MQTT': [1883, 8883],
            'Is_AMQP': 5672,
            'Is_Redis': 6379,
            'Is_Memcached': 11211,
            'Is_Elasticsearch': [9200, 9300],
            'Is_Zookeeper': 2181,
            'Is_Cassandra': 9042,
            'Is_Docker': 2375,
            'Is_Kubernetes': 6443,
            'Is_SMB_Direct': 445, # Already present
            'Is_iSCSI': 3260,
            'Is_AFP': 548,
            'Is_DHCPv6': [546, 547],
            'Is_RIPng': 521,
            'Is_OSPF': 89,
            'Is_PPPoE': 3544,
            'Is_L2TP': 1701,
            'Is_GRE': 47, # Protocol number, not TCP/UDP port
            'Is_ESP': 50, # Protocol number
            'Is_AH': 51  # Protocol number
            # Add even more as needed
        }

        # Create the binary categorical features
        for protocol, ports in protocol_ports_extensive.items():
            if isinstance(ports, int):
                df[protocol] = (df['Port number'] == ports)
            elif isinstance(ports, list):
                df[protocol] = df['Port number'].isin(ports)
            elif isinstance(ports, range):
                df[protocol] = df['Port number'].isin(ports)
            else:
                print(f"Warning: Unexpected port type for {protocol}")

        return df
    
    def add_features_train(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_features(df)

        self.pca_train(df)
        df = self.pca_transform(df, self.pca)

        self.kmeans_train(df)
        df = self.kmeans_clustering(df)

        self.umap_train(df)
        df = self.umap_embedding(df)

        df = self.oversample(df)

        return df
    
    def add_features_test(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_features(df)

        df = self.pca_transform(df, self.pca)
        df = self.kmeans_clustering(df)
        df = self.umap_embedding(df)
        return df
    
    pca = PCA(n_components=5, random_state=42)

    @time_it
    def pca_train(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("PCA tanítás...")
        x_cols = [col for col in df.columns if col not in ['Type', 'Start time', 'End time'] and not df[col].isna().any() ]

        self.pca.fit(df[x_cols])
    
    @time_it
    def pca_transform(self, df: pd.DataFrame, pca) -> pd.DataFrame:
        logger.info("PCA transzformáció...")
        x_cols = [col for col in df.columns if col not in ['Type', 'Start time', 'End time'] and not df[col].isna().any() ]
        pca_features = self.pca.transform(df[x_cols])
        pca_cols = [f'PCA_{i+1}' for i in range(self.pca.n_components_)]
        df[pca_cols] = pca_features
        return df
    
    kmeans_20 = KMeans(n_clusters=20, random_state=42, n_init=10)
    
    kmeans_114 = KMeans(n_clusters=114, random_state=42, n_init=10)

    @time_it
    def kmeans_train(self, df: pd.DataFrame):
        logger.info("KMeans tanítás...")
        x_cols = [col for col in df.columns if col not in ['Type', 'Start time', 'End time'] and not df[col].isna().any()]
        self.kmeans_20.fit(df[x_cols])
        self.kmeans_114.fit(df[x_cols])
        

    @time_it
    def kmeans_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("KMeans klaszterezés...")
        x_cols = [col for col in df.columns if col not in ['Type', 'Start time', 'End time'] and not df[col].isna().any()]
        
        df['cluster'] = self.kmeans_20.labels_
        df_dists = self.kmeans_20.transform(df[x_cols])
        dist_cols = [f'dist_centroid_{i}' for i in range(self.kmeans_20.n_clusters)]
        df[dist_cols] = df_dists

        df['cluster'] = self.kmeans_114.labels_
        df_dists = self.kmeans_114.transform(df[x_cols])
        dist_cols = [f'dist_centroid_{i}' for i in range(self.kmeans_114.n_clusters)]
        df[dist_cols] = df_dists

        return df
    
    reducer = UMAP(n_components=2, n_neighbors=30, n_jobs=-1, verbose=True, low_memory=True)

    @time_it
    def umap_train(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("UMAP tanítás...")
        x_cols = [col for col in df.columns if col not in ['Type', 'Start time', 'End time'] and not df[col].isna().any()]

        self.reducer.fit(df[x_cols])
    
    @time_it
    def umap_embedding(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("UMAP beágyazás...")
        x_cols = [col for col in df.columns if col not in ['Type', 'Start time', 'End time'] and not df[col].isna().any()]

        embedding = self.reducer.transform(df[x_cols])

        df[['umap_1', 'umap_2']] = embedding

        return df

    @time_it
    def oversample(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Oversampling...")

        class_counts = df['Type'].value_counts()

        sampling_strategy = {}
        majority_class_count = class_counts.max()
        for cls, count in class_counts.items():
            if count < majority_class_count:
                sampling_strategy[cls] = int(count * 1.5)


        x_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in ['Type', 'Start time', 'End time']]
        smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(df[x_cols], df['Type'])
        return pd.concat([X_resampled, y_resampled], axis=1)
    
class ModelHandler:
    def __init__(self):
        pass

    x_rfc = ['Min_PacketSpeed','Std_DataSpeed','Std_DetectCount','Mean_DetectCount','Avg source IP count','Max_DetectCount','Attack ID','Max_PacketSpeed','total_seconds','AvgPacketLen_Std','Mean_DataSpeed','Mean_PacketSpeed','Source IP count','Cos_DayOfYear','Sin_DayOfYear','DayOfYear','AvgPacketLen_Mean','Min_DataSpeed','PCA_4','Start Hour']
    
    x_xgb = ['dist_centroid_silhouette_72','Avg source IP count','PCA_4','dist_centroid_silhouette_61','PCA_3',
    'Std_DetectCount',
    'Is_SSH',
    'dist_centroid_silhouette_54',
    'Std_DataSpeed',
    'umap_2',
    'Min_PacketSpeed',
    'Data speed',
    'Mean_DetectCount',
    'Victim IP',
    'Min_DataSpeed',
    'AvgPacketLen_Mean',
    'Attack ID',
    'PacketSpeed_Per_Second',
    'PCA_2',
    'Mean_PacketSpeed']

    x_lgbm = ['total_seconds',
    'AvgPacketLen_Mean',
    'Avg source IP count',
    'Max_DetectCount',
    'Std_DataSpeed',
    'AvgPacketLen_Std',
    'Min_PacketSpeed',
    'Sin_DayOfYear',
    'Cos_DayOfYear',
    'Mean_DataSpeed',
    'Max_DataSpeed',
    'Mean_PacketSpeed',
    'Attack ID',
    'Max_PacketSpeed',
    'Victim IP',
    'AvgPacketLen_Per_TotalSeconds',
    'Mean_DetectCount',
    'Start Hour',
    'Min_DataSpeed',
    'DayOfYear',
    'VictimIP_Count',
    'Std_DetectCount',
    'PCA_3',
    'PCA_4',
    'PacketSpeed_Per_Second',
    'Source IP count',
    'PCA_2',
    'umap_2',
    'Is_SSH',
    'dist_centroid_silhouette_54']

    base_models = [
        XGBClassifier(
        objective='multi:softmax',
            num_class=3,
            booster=('dart'),
            alpha=0.6730647547618864,
            subsample=0.42803065237564275,
            colsample_bytree=0.8565965677640711,
            max_depth=5,
            eta=0.179806132737519,
            gamma=0.8114340643087419,
            grow_policy='lossguide',
            min_child_weight=5,
            eval_metric='merror',
            random_state=42
            ),
        RandomForestClassifier(n_estimators=287, max_depth=22, min_samples_split=4, min_samples_leaf=5, random_state=42, class_weight='balanced'),
        lgb.LGBMClassifier(verbose=-1, n_estimators=441, learning_rate=0.0895186049880042, num_leaves=98, max_depth=8, 
                            min_child_samples=34, subsample=0.7024962299198043, colsample_bytree=0.8199213047640314, random_state=42, is_unbalance=True),



        DecisionTreeClassifier(max_depth=4, min_samples_leaf=819, criterion='gini', random_state=42)  # Meta-learner can be any classifier; Logistic Regression is a common choice
    ]
    meta_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=819, criterion='gini', random_state=42) # The final model that blends them all

    @time_it
    def manual_stacking_fit(self, train, n_splits=5):
        logger.info("Stacking model tanítása...")

        X_train_list = [train[self.x_xgb], train[self.x_rfc], train[self.x_lgbm]]

        n_samples_train = X_train_list[0].shape[0]
        n_classes = len(np.unique(train['Type']))
        n_models = len(self.base_models)
        
        # Initialize empty arrays to hold the "meta-features" (predicted probabilities)
        # Total columns = (number of base models) * (number of classes)
        oof_features = np.zeros((n_samples_train, n_models * n_classes))
        
        # Use Stratified K-Fold to generate out-of-fold predictions safely
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_list[0], train['Type'])):
            
            # Train each base model on its specific slice of this fold's training data
            for m_idx, (model, X_base_train) in enumerate(zip(self.base_models, X_train_list)):
                # Clone ensures we train a fresh copy of the model configuration on every fold
                fold_model = clone(model) 
                
                # Fit on fold training data
                fold_model.fit(X_base_train[train_idx], train['Type'][train_idx])
                
                # Calculate the feature columns index for this specific model
                col_start = m_idx * n_classes
                col_end = col_start + n_classes
                
                # 1. Populate training meta-features using the validation slice (Out-of-Fold)
                oof_features[val_idx, col_start:col_end] = fold_model.predict_proba(X_base_train[val_idx])
        
        # 4. Train the meta-model on the out-of-fold features
        self.meta_model.fit(oof_features, train['Type'])

        for model, X_base_train in zip(self.base_models, X_train_list):
            model.fit(X_base_train, train['Type'])

    @time_it
    def predict(self, test):
        logger.info("Predikció...")
        X_test_list = [test[self.x_xgb], test[self.x_rfc], test[self.x_lgbm]]
        probs_list = [model.predict_proba(X) for model, X in zip(self.base_models, X_test_list)]
        test_probs_stacked = np.hstack(probs_list)
        final_predictions = self.meta_model.predict(test_probs_stacked)
    
        logger.info("Predikciós eredmények...")
        logger.info(classification_report(test['Type'], final_predictions))


def main():
    logger.info("DDoS Detekciós Pipeline indítása...")
    
    # 1. Komponensek inicializálása
    loader = DataLoader()
    preprocessor = Preprocessor()
    model_handler = ModelHandler()
    
    # 2. Pipeline végrehajtása
    try:
        train_data = loader.load_train()
        processed_train = preprocessor.process(train_data)
        train = preprocessor.add_features_train(processed_train)
        model_handler.manual_stacking_fit(train)
        
        logger.info("Tanító pipeline sikeresen lefutott. Kiértékelő adatok feldolgozása...")

        test_data = loader.load_test("C")
        processed_test = preprocessor.process(test_data)
        test = preprocessor.add_features_test(processed_test)
        model_handler.predict(test)
        
        logger.info("Pipeline sikeresen lefutott.")
    except Exception as e:
        logger.error(f"Kritikus hiba a pipeline futása közben: {str(e)}")

if __name__ == "__main__":
    # Parancssori argumentumok kezelése
    parser = argparse.ArgumentParser(description="DDoS Detection Pipeline")
    args = parser.parse_args()
    
    main()