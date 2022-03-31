
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=(28, 28, 3),
                 test=False):
        
        
        self.df = df.copy() #[i|img,label,use] use here is a random index
        self.X_col = X_col #img
        self.y_col = y_col #label
        self.batch_size = batch_size
        self.input_size = input_size
        self.test = test
        #add the other key vars here
        #also add stats holding info
        self.dataused = len(self.df.index)
        #TODO
        self.n = len(self.df)
        self.n_name = df[y_col['name']].nunique()
        self.n_type = df[y_col['type']].nunique()
    
    def on_epoch_end(self,use):
        #use = [i|use] where use is from 0 to dataused and nan past dataused
        #update the used col in df
        self.df['use'] = use
        #sort the large dataframe by the use index and put nans last so not used
        self.df = self.df.sort_values('use',na_position='last')
        self.dataused = pd.max(self.df['use'])
    
    def __get_input(self, path, bbox, target_size):
        #TODO
        xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        #TODO
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        #TODO
        path_batch = batches[self.X_col['path']]
        bbox_batch = batches[self.X_col['bbox']]
        
        name_batch = batches[self.y_col['name']]
        type_batch = batches[self.y_col['type']]

        X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        return X_batch, tuple([y0_batch, y1_batch])
    
    def __getitem__(self, index):
        dataused = pd.max(df['use'])
        batches = self.df[index * self.batch_size:(index+1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.dataused // self.batch_size
