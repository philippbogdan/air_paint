                                                                           
from capture.sync import list_available_cameras, get_camera_names, auto_select_cameras, is_iphone_camera                                                                      
                                                                                       
print('=== Available camera indices (OpenCV) ===')                                     
available = list_available_cameras()                                                   
print(available)                                                                       
                                                                                       
print('\n=== Camera names (system_profiler) ===')                                      
names = get_camera_names()                                                             
for idx, name in names.items():                                                        
    is_iphone = is_iphone_camera(name)                                                 
    print(f'  {idx}: \"{name}\" -> is_iphone={is_iphone}')                             
                                                                                       
print('\n=== Auto-selected cameras ===')                                               
cam_a, cam_b = auto_select_cameras()                                                   
print(f'Camera A: {cam_a} ({names.get(cam_a, "Unknown")})')                          
print(f'Camera B: {cam_b} ({names.get(cam_b, "Unknown")})')                          
