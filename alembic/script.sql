INSERT INTO users (name, email) VALUES ('User1', 'user1@example.com');
INSERT INTO users (name, email) VALUES ('User2', 'user2@example.com');


INSERT INTO directories (name, description, parent_id) VALUES ('RaDICaL', 'radical description', NULL); 
INSERT INTO directories (name, description, parent_id) VALUES ('radar_high_res', 'radar high res description', 1);
INSERT INTO directories (name, description, parent_id) VALUES ('indoor_human', 'indoor human description', 1);
INSERT INTO directories (name, description, parent_id) VALUES ('30m_collection', '30m_collection description', 1);
INSERT INTO directories (name, description, parent_id) VALUES ('50m_collection', '50m_collection description', 1);

INSERT INTO directories (name, description, parent_id) VALUES ('RADIal', 'radial description', NULL);


INSERT INTO files (name, description, size, path, directory_id) VALUES ('1.bag', 'description', '2.5GB', './1.bag', 2);
INSERT INTO files (name, description, size, path, directory_id) VALUES ('2.bag', 'description', '2.5GB', './2.bag', 2);
INSERT INTO files (name, description, size, path, directory_id) VALUES ('3.bag', 'description', '2.5GB', './3.bag', 2);

INSERT INTO files (name, description, size, path, directory_id) VALUES ('2020-03-05-16-35-19.bag', 'description', '2.5GB', 'dataset/indoor_human', 3);
INSERT INTO files (name, description, size, path, directory_id) VALUES ('2020-03-06-15-00-49.bag', 'description', '2.5GB', 'dataset/indoor_human', 3);
INSERT INTO files (name, description, size, path, directory_id) VALUES ('2020-03-10-15-55-08.bag', 'description', '2.5GB', 'dataset/indoor_human', 3);

INSERT INTO files (name, description, size, path, directory_id) VALUES ('4.bag', 'description', '2.5GB', './4.bag', 4);
INSERT INTO files (name, description, size, path, directory_id) VALUES ('5.bag', 'description', '2.5GB', './5.bag', 4);
INSERT INTO files (name, description, size, path, directory_id) VALUES ('6.bag', 'description', '2.5GB', './6.bag', 4);

INSERT INTO files (name, description, size, path, directory_id) VALUES ('7.bag', 'description', '2.5GB', './7.bag', 5);
INSERT INTO files (name, description, size, path, directory_id) VALUES ('8.bag', 'description', '2.5GB', './8.bag', 5);
INSERT INTO files (name, description, size, path, directory_id) VALUES ('9.bag', 'description', '2.5GB', './9.bag', 5);

INSERT INTO files (name, description, size, path, directory_id) VALUES ('RECORD@2020-11-22_12.54.38.zip', 'description', '2.5GB', 'dataset/radial',6);
INSERT INTO files (name, description, size, path, directory_id) VALUES ('RECORD@2020-11-22_12.48.07.zip', 'description', '2.5GB', 'dataset/radial',6);