INSERT INTO users (name, email) VALUES ('User1', 'user1@example.com');
INSERT INTO users (name, email) VALUES ('User2', 'user2@example.com');


INSERT INTO directories (name, description, parent_id) VALUES ('RaDICaL', 'radical description', NULL); 
INSERT INTO directories (name, description, parent_id) VALUES ('radar_high_res', 'radar high res description', 1);
INSERT INTO directories (name, description, parent_id) VALUES ('indoor_human', 'indoor human description', 1);
INSERT INTO directories (name, description, parent_id) VALUES ('30m_collection', '30m_collection description', 1);
INSERT INTO directories (name, description, parent_id) VALUES ('50m_collection', '50m_collection description', 1);

INSERT INTO directories (name, description, parent_id) VALUES ('RADIal(raw)', 'RADIal raw data record, only part labeled', NULL);

INSERT INTO directories (name, description, parent_id) VALUES ('RADIal(labeled)', 'RADIal ready-to-use labeled data, 8252 frames in total', NULL);

INSERT INTO directories (name, description, parent_id) VALUES ('CRUW_ROD2021', 'CRUW description', NULL);

INSERT INTO directories (name, description, parent_id) VALUES ('CARRADA', 'CARRADA description', NULL);

INSERT INTO directories (name, description, parent_id) VALUES ('RADDet_dataset', 'RADDet description', NULL);

INSERT INTO directories (name, description, parent_id) VALUES ('Astyx', 'Astyx description', NULL);

INSERT INTO directories (name, description, parent_id) VALUES ('K-Radar', 'description', NULL);


INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('1.bag', 'description', '2.5GB', './1.bag', 2, '', '', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2.bag', 'description', '2.5GB', './2.bag', 2, '', '', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('3.bag', 'description', '2.5GB', './3.bag', 2, '', '', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE);
                                                                               
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-03-05-16-35-19.bag', 'description', '44GB', 'radical/indoor_human', 3, 'radical/radarcfg', 'RaDICaL', TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-04-10-13-37-01.bag', 'description', '3.8GB', 'radical/indoor_human', 3, 'radical/radarcfg', 'RaDICaL', TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-03-10-15-55-08.bag', 'description', '57MB', 'radical/indoor_human', 3, 'radical/radarcfg', 'RaDICaL', TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE);
                                                                               
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('4.bag', 'description', '2.5GB', './4.bag', 4, '', '', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('5.bag', 'description', '2.5GB', './5.bag', 4, '', '', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('6.bag', 'description', '2.5GB', './6.bag', 4, '', '', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE);
                                                                               
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('7.bag', 'description', '2.5GB', './7.bag', 5, '', '', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('8.bag', 'description', '2.5GB', './8.bag', 5, '', '', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('9.bag', 'description', '2.5GB', './9.bag', 5, '', '', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE);
                                                                               
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('RECORD@2020-11-22_12.54.38.zip', 'description', '2.5GB', 'radial',6, '', 'RADIal', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('RECORD@2020-11-22_12.48.07.zip', 'description', '2.5GB', 'radial',6, '', 'RADIal', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE);

INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('RADIal-labeled', 'description', '120GB', '/home/kangle/dataset/radial',7, '', 'RADIal', FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE);

INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019_04_09_BMS1000', 'description', '463MB', '/home/kangle/dataset/CRUW', 8, '/home/kangle/dataset/CRUW/sensor_config_rod2021.json', 'CRUW', FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019_04_09_BMS1000', 'description', '463MB', '/home/kangle/dataset/CRUW', 8, '/home/kangle/dataset/CRUW/sensor_config_rod2021.json', 'CRUW', FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019_04_09_BMS1002', 'description', '463MB', '/home/kangle/dataset/CRUW', 8, '/home/kangle/dataset/CRUW/sensor_config_rod2021.json', 'CRUW', FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019_04_09_CMS1002', 'description', '463MB', '/home/kangle/dataset/CRUW', 8, '/home/kangle/dataset/CRUW/sensor_config_rod2021.json', 'CRUW', FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);

INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-12-52-12', 'description', '7.4G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-12-55-51', 'description', '4.6G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-12-58-42', 'description', '6.7G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-13-03-38', 'description', '6.1G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-13-06-41', 'description', '5.1G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-13-11-12', 'description', '2.8G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-13-13-01', 'description', '2.0G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-13-14-29', 'description', '3.3G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-13-18-33', 'description', '2.0G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-13-20-20', 'description', '1.9G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-13-23-22', 'description', '2.2G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2019-09-16-13-25-35', 'description', '2.6G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-12-12-16', 'description', '3.5G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-12-13-54', 'description', '4.7G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-12-16-05', 'description', '5.5G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-12-17-57', 'description', '4.4G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-12-20-22', 'description', '2.8G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-12-22-05', 'description', '2.7G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-12-23-30', 'description', '2.8G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-05-44', 'description', '1.5G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-06-53', 'description', '1.7G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-07-38', 'description', '2.0G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-08-51', 'description', '1.8G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-09-58', 'description', '1.7G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-10-51', 'description', '1.8G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-11-45', 'description', '1.2G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-12-42', 'description', '2.2G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-13-43', 'description', '1.6G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-14-35', 'description', '2.2G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);
INSERT INTO files (name, description, size, path, directory_id, config, parse, "ADC", "RAD", "RA", "AD", "RD", spectrogram, "radarPC", "lidarPC", image, depth_image, labeled) VALUES ('2020-02-28-13-15-36', 'description', '1.2G', '/home/kangle/Downloads/wd_disk_radar_data/CARRADA/Carrada', 9, '', 'CARRADA', FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE);

