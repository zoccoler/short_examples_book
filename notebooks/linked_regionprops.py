def mode(regionmask, intensity_image):
    '''Returns the most frequent element from a masked image.'''
    import numpy as np
    return np.bincount(np.ravel(intensity_image[regionmask].astype(int))).argmax()

def make_element_wise_dict(a,b):
    '''Returns an element-wise dictionary of keys from a and values from b'''
    return dict(list(map(lambda a, b: [a,b], *[a, b])))

def linked_regionprops_table(labels_array, intensity_image, properties, ref_channel = 0, **kwargs):
    '''Measure properties from 2 channels in a linked way.
    
    For each channel in a multi-channel image and a multi-channel label array, 
    it measures properties of a reference channel (default: 0) and a probe channel. 
    For each measurement of an object in the probe channel, it returns the same measurement of the corresponding object in the reference channel.
    Corresponding object here means the underlying label from the reference channel with the highest overlap to the label from the probe channel.
    '''
    from skimage import measure
    from pandas import DataFrame
    import numpy as np
    import pandas as pd
     
    n_channels = intensity_image.shape[-1]
    table_list = []
    cols = []
    # Measure properties of reference channel
    ref_channel_props =  measure.regionprops_table(label_image = labels_array[...,ref_channel],
                                                intensity_image = intensity_image[...,ref_channel],
                                                properties = ['label'] + properties,
                                                **kwargs,
                                               )
    
    for i in range(n_channels):
        if i != ref_channel:
            # Create label links from probe channel to reference channel
            label_links= pd.DataFrame(
                measure.regionprops_table(label_image = labels_array[...,i], 
                                          intensity_image = labels_array[...,ref_channel], 
                                          properties = ['label',],
                                          extra_properties = [mode]
                                         )
            ).astype(int)
            # rename column
            label_links.rename(columns={'label':'label-ch' + str(i), 'mode':'label-of-obj-at-ch' + str(ref_channel)}, inplace=True)
            
            
            
            # Include extra properties of reference channel
            properties_with_extras = [props for props in ref_channel_props if props != 'label']
            for props in properties_with_extras:
                props_mapping = make_element_wise_dict(ref_channel_props['label'].tolist(), ref_channel_props[props].tolist())
                # label_links.insert(1, props + '-of-obj-at-ref-ch' + str(ref_channel), label_links['label-of-obj-at-ref-ch' + str(ref_channel)])
                label_links[props + '-of-obj-at-ch' + str(ref_channel)] = label_links['label-of-obj-at-ch' + str(ref_channel)]
                label_links = label_links.replace({props + '-of-obj-at-ch' + str(ref_channel) : props_mapping})

            col_names = label_links.columns.to_list()
            # Append properties of probe channel
            probe_channel_props = pd.DataFrame(
                measure.regionprops_table(label_image = labels_array[...,i], 
                                          intensity_image = intensity_image[...,i], 
                                          properties = properties,
                                          **kwargs,
                                         )
            )
            # rename column
            probe_channel_props.rename(columns=dict([(props , props + '-ch' + str(i)) for props in properties_with_extras]), inplace=True)
            table = pd.concat([label_links, probe_channel_props], axis=1)
            col_names[1:1] = probe_channel_props.columns.to_list()
            table = table[col_names]
            
            table_list += [table]
        
    return table_list