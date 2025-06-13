
# Ejemplo de uso 

import math
  import random
  import json
  import os
  
  # Funciones auxiliares
  def contar(lista):
      conteo = {}
      for item in lista:
          if item in conteo:
              conteo[item] += 1
          else:
              conteo[item] = 1
      return conteo
  
  def mayoritario(lista):
      conteo = contar(lista)
      max_cuenta = -1
      clase = None
      for k in conteo:
          if conteo[k] > max_cuenta:
              max_cuenta = conteo[k]
              clase = k
      return clase
  
  # Clase DecisionTree
  class DecisionTree:
      def __init__(self, max_depth=5, min_samples_split=2):
          self.max_depth = max_depth
          self.min_samples_split = min_samples_split
          self.tree = None
  
      def fit(self, X, y):
          self.tree = self._build_tree(X, y)
  
      def predict(self, X):
          return [self._predict_single(x, self.tree) for x in X]
  
      def _build_tree(self, X, y, depth=0):
          n_samples = len(X)
          n_features = len(X[0]) if n_samples > 0 else 0
  
          if (depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) == 1):
              return {'class': mayoritario(y)}
  
          best_feature, best_threshold = self._best_split(X, y)
  
          if best_feature is None:
              return {'class': mayoritario(y)}
  
          left_X, left_y, right_X, right_y = [], [], [], []
          for i in range(n_samples):
              if X[i][best_feature] <= best_threshold:
                  left_X.append(X[i])
                  left_y.append(y[i])
              else:
                  right_X.append(X[i])
                  right_y.append(y[i])
  
          left_subtree = self._build_tree(left_X, left_y, depth + 1)
          right_subtree = self._build_tree(right_X, right_y, depth + 1)
  
          return {
              'feature': best_feature,
              'threshold': best_threshold,
              'left': left_subtree,
              'right': right_subtree
          }
  
      def _best_split(self, X, y):
          best_gini = 1
          best_feature = None
          best_threshold = None
  
          n_features = len(X[0]) if len(X) > 0 else 0
          m = min(int(math.sqrt(n_features)) + 1, n_features)
  
          features = []
          while len(features) < m:
              f = random.getrandbits(8) % n_features
              if f not in features:
                  features.append(f)
  
          for feature in features:
              thresholds = []
              for i in range(len(X)):
                  val = X[i][feature]
                  if val not in thresholds:
                      thresholds.append(val)
  
              for threshold in thresholds:
                  left_y, right_y = [], []
                  for i in range(len(X)):
                      if X[i][feature] <= threshold:
                          left_y.append(y[i])
                      else:
                          right_y.append(y[i])
  
                  if not left_y or not right_y:
                      continue
  
                  gini = self._gini_index(left_y, right_y)
                  if gini < best_gini:
                      best_gini = gini
                      best_feature = feature
                      best_threshold = threshold
  
          return best_feature, best_threshold
  
      def _gini_index(self, left_y, right_y):
          n_left = len(left_y)
          n_right = len(right_y)
          n_total = n_left + n_right
  
          def gini(y_list):
              clases = set(y_list)
              score = 1.0
              for c in clases:
                  p = y_list.count(c) / len(y_list)
                  score -= p * p
              return score
  
          return (n_left / n_total) * gini(left_y) + (n_right / n_total) * gini(right_y)
  
      def _predict_single(self, x, node):
          if 'class' in node:
              return node['class']
          if x[node['feature']] <= node['threshold']:
              return self._predict_single(x, node['left'])
          else:
              return self._predict_single(x, node['right'])
  
  # Clase RandomForest (completamente implementada)
  class RandomForest:
      def __init__(self, n_estimators=5, max_depth=5, min_samples_split=2):
          self.n_estimators = n_estimators
          self.max_depth = max_depth
          self.min_samples_split = min_samples_split
          self.trees = []
  
      def fit(self, X, y):
          self.trees = []
          n_samples = len(X)
          for _ in range(self.n_estimators):
              indices = []
              for _ in range(n_samples):
                  indices.append(random.getrandbits(8) % n_samples)
              X_sample = [X[i] for i in indices]
              y_sample = [y[i] for i in indices]
              tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
              tree.fit(X_sample, y_sample)
              self.trees.append(tree)
  
      def predict(self, X):
          predictions = []
          for x in X:
              votes = []
              for tree in self.trees:
                  vote = tree.predict([x])[0]
                  votes.append(vote)
              predictions.append(mayoritario(votes))
          return predictions
  
  # Simulación de sensores
  class SensorSimulator:
      @staticmethod
      def analog_read():
          return random.randint(0, 1023)
  
      @staticmethod
      def digital_read():
          return random.choice([0, 1])
  
  # Gestor de Sensores
  class SensorManager:
      def __init__(self):
          self.sensors = {}
          self.data = []
          self.labels = []
          self.model = RandomForest(n_estimators=5, max_depth=5)
  
      def add_sensor(self, name, sensor_type):
          self.sensors[name] = sensor_type
  
      def read_sensors(self):
          readings = {}
          for name, sensor_type in self.sensors.items():
              if sensor_type == "analog":
                  readings[name] = SensorSimulator.analog_read()
              elif sensor_type == "digital":
                  readings[name] = SensorSimulator.digital_read()
          return readings
  
      def save_to_file(self, filename="sensor_data.json"):
          data = {
              "X": self.data,
              "y": self.labels
          }
          with open(filename, "w") as f:
              json.dump(data, f)
  
      def load_from_file(self, filename="sensor_data.json"):
          try:
              with open(filename, "r") as f:
                  data = json.load(f)
              self.data = data["X"]
              self.labels = data["y"]
              return True
          except:
              return False
  
      def train_model(self):
          if len(self.data) > 0 and len(self.labels) > 0:
              self.model.fit(self.data, self.labels)
              return True
          return False
  
      def predict(self, sensor_readings=None):
          if sensor_readings is None:
              sensor_readings = self.read_sensors()
  
          features = [sensor_readings[name] for name in sorted(self.sensors.keys())]
          return self.model.predict([features])[0]
  
      def collect_data(self, label):
          readings = self.read_sensors()
          self.data.append([readings[name] for name in sorted(self.sensors.keys())])
          self.labels.append(label)
  
  # Ejemplo de uso
  if __name__ == '__main__':
      print("Sistema de Predicción con Sensores - Demo")
      print("----------------------------------------")
  
      # Configurar sistema
      sistema = SensorManager()
      sistema.add_sensor("temperatura", "analog")
      sistema.add_sensor("humedad", "analog")
      sistema.add_sensor("movimiento", "digital")
  
      # Recolectar datos de entrenamiento
      print("\nEntrenando modelo con datos simulados...")
      for i in range(100):
          if i < 70:  # 70% datos normales
              sistema.collect_data(0)
          else:       # 30% datos de alerta
              sistema.collect_data(1)
  
      # Entrenar modelo
      sistema.train_model()
  
      # Probar el sistema
      print("\nProbando predicciones:")
      for _ in range(5):
          lectura = sistema.read_sensors()
          prediccion = sistema.predict(lectura)
          estado = "Normal" if prediccion == 0 else "ALERTA"
          print(f"Lectura: {lectura} -> Estado: {estado}")
  
      print("\nDemo completada exitosamente!")
      print("")
      print("Jannet Ortiz Aguilar")

# RESULTADOS

Sistema de Predicción con Sensores - Demo
----------------------------------------

Entrenando modelo con datos simulados...

Probando predicciones:
Lectura: {'temperatura': 674, 'humedad': 114, 'movimiento': 0} -> Estado: Normal
Lectura: {'temperatura': 512, 'humedad': 761, 'movimiento': 0} -> Estado: Normal
Lectura: {'temperatura': 905, 'humedad': 712, 'movimiento': 0} -> Estado: Normal
Lectura: {'temperatura': 678, 'humedad': 553, 'movimiento': 0} -> Estado: Normal
Lectura: {'temperatura': 504, 'humedad': 759, 'movimiento': 1} -> Estado: Normal

Demo completada exitosamente!

Jannet Ortiz Aguilar
