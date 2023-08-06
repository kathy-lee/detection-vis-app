INSERT INTO users (name, email) VALUES ('User1', 'user1@example.com');
INSERT INTO users (name, email) VALUES ('User2', 'user2@example.com');


INSERT INTO directories (name, description, parent_id) VALUES ('RaDICaL', 'radical description', NULL); 
INSERT INTO directories (name, description, parent_id) VALUES ('radar_high_res', 'radar high res description', 1);
INSERT INTO directories (name, description, parent_id) VALUES ('indoor_human', 'indoor human description', 1);
INSERT INTO directories (name, description, parent_id) VALUES ('30m_collection', '30m_collection description', 1);
INSERT INTO directories (name, description, parent_id) VALUES ('50m_collection', '50m_collection description', 1);

INSERT INTO directories (name, description, parent_id) VALUES ('RADIal(raw)', 'RADIal raw data record, only part labeled', NULL);

INSERT INTO directories (name, description, parent_id) VALUES ('RADIal(labeled)', 'RADIal ready-to-use labeled data, 8252 frames in total', NULL);



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

