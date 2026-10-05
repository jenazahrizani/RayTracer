Project ini sudah jauh lebih serius daripada sekadar demo ray tracer + AI. Kalau targetmu adalah membuatnya terasa seperti **AI navigation research lab yang benar-benar matang**, saya akan mengejar beberapa hal berikut, dalam urutan prioritas.

## 1. Buat training benar-benar reproducible

Sekarang model bisa disimpan, tetapi yang paling ideal adalah **checkpoint training lengkap**, bukan hanya bobot network.

Simpan:

```text
CHECKPOINT
├── online network
├── target network
├── replay buffer
├── epsilon
├── learning rate
├── gamma
├── training step
├── episode
├── RNG state
├── environment seed
├── difficulty
└── obstacle configuration
```

Dengan begitu:

```text
TRAIN 500000
↓
SAVE CHECKPOINT
↓
browser ditutup
↓
LOAD CHECKPOINT
↓
TRAIN 500000 lagi
```

dan training benar-benar dilanjutkan, bukan sekadar memuat bobot.

Ini menurut saya **fitur nomor satu** yang perlu ditambahkan.

---

## 2. Tambahkan curriculum learning

Saat ini environment bisa dibuat sulit, tetapi AI sebaiknya **tidak langsung dilempar ke difficulty tinggi**.

Contohnya:

```text
LEVEL 1
4 obstacles
target dekat
tidak banyak dead-end

        ↓

LEVEL 2
8 obstacles
target lebih jauh

        ↓

LEVEL 3
12 obstacles
obstacle lebih rapat

        ↓

LEVEL 4
random maze

        ↓

LEVEL 5
hard navigation
```

AI baru naik level kalau success rate misalnya:

```text
> 85% selama 100 episode
```

Ini jauh lebih bagus daripada:

```text
TRAIN 100000
difficulty 3
```

terus berharap neural network menemukan semuanya sendiri.

---

## 3. Prioritized Experience Replay

Sekarang replay buffer mengambil pengalaman secara relatif biasa.

Upgrade besar berikutnya:

**Prioritized Experience Replay (PER).**

Transition yang mengejutkan network akan lebih sering dipelajari.

Misalnya:

```text
normal:
reward +0.01
TD error kecil

collision:
reward -1
TD error besar
```

Collision tersebut menjadi jauh lebih penting untuk dipelajari.

Strukturnya:

```text
Replay Buffer

experience A   priority 0.1
experience B   priority 0.3
experience C   priority 8.7   ← collision
experience D   priority 0.2
```

Ini sangat cocok untuk navigation.

---

## 4. Multi-step DQN

Tambahkan **n-step return**, misalnya 3-step atau 5-step.

Daripada belajar:

```text
s0 → a0 → r0 → s1
```

agent bisa belajar:

```text
s0
 ↓
a0
 ↓
r0
 ↓
a1
 ↓
r1
 ↓
a2
 ↓
r2
```

Dengan demikian reward seperti:

```text
TARGET REACHED
```

bisa lebih cepat disebarkan ke action-action sebelumnya.

Untuk environment navigation, ini biasanya sangat berguna karena target bisa berjarak puluhan langkah.

---

## 5. Distributional / Dueling DQN

Tahap berikutnya bisa membuat network:

```text
37
 ↓
128
 ↓
128
 ↓
 ┌───────────────┐
 │               │
Value stream   Advantage stream
 │               │
 └───────┬───────┘
         ↓
      Q-values
```

Ini disebut **Dueling DQN**.

AI bisa membedakan:

```text
"posisi ini buruk"
```

dan:

```text
"di posisi ini LEFT lebih baik daripada RIGHT"
```

secara lebih jelas.

Kalau ingin melangkah lebih jauh:

**Rainbow DQN**.

Rainbow menggabungkan beberapa teknik seperti:

```text
Double DQN
+
Dueling Network
+
Prioritized Replay
+
Multi-step learning
+
Noisy Networks
+
Distributional RL
```

Tapi saya **tidak akan langsung lompat ke Rainbow**. Bangun satu-satu supaya telemetry tetap bisa menjelaskan apa yang terjadi.

---

# 6. Buat AI benar-benar punya “memory”

Sekarang network menerima state saat ini.

Upgrade menarik:

```text
state(t-3)
state(t-2)
state(t-1)
state(t)
```

kemudian gunakan:

```text
MLP
```

atau:

```text
GRU / LSTM
```

Jadi agent bisa mengetahui:

> “Saya baru saja berbelok kiri dan ternyata buntu.”

Ini sangat berguna kalau environment nanti dibuat lebih kompleks atau partially observable.

---

# 7. Sensor ray dibuat lebih cerdas

Sekarang kamu sudah punya 32 ray, itu bagus.

Saya akan mengembangkan menjadi:

```text
32 ray distance
+
32 ray relative velocity
+
target direction
+
target distance
+
current speed
+
angular velocity
+
last action
```

Misalnya:

```text
RAY 00 = 0.91
RAY 01 = 0.84
...
RAY 31 = 0.12

TARGET ANGLE = -0.42
TARGET DIST = 0.73
SPEED = 0.51
LAST ACTION = RIGHT
```

Ini membuat state jauh lebih informatif.

---

# 8. Action space jangan terlalu sederhana selamanya

Sekarang:

```text
FORWARD
LEFT
RIGHT
BRAKE
REVERSE
```

Ini bagus untuk bootstrap.

Tahap lebih lanjut bisa membuat:

```text
steering ∈ [-1, 1]
throttle ∈ [-1, 1]
```

Jadi bukan lagi discrete action:

```text
LEFT
RIGHT
FORWARD
```

tetapi continuous control.

Untuk itu model bisa berkembang menuju:

```text
DDPG
TD3
SAC
```

Tetapi ini saya taruh **jauh setelah DQN matang**.

---

# 9. Buat benchmark otomatis

Ini menurut saya salah satu fitur paling penting agar project terasa seperti **research project**, bukan sekadar visual demo.

Misalnya command:

```text
BENCH AI
```

menghasilkan:

```text
=========================================================
RAYTRC.AI / BENCHMARK
=========================================================

EPISODES                 1000
SUCCESS                  873
COLLISION                97
TIMEOUT                  30

SUCCESS RATE             87.30%
COLLISION RATE            9.70%

AVG STEPS                148.3
AVG REWARD                42.17

P50 STEPS                131
P95 STEPS                284

GENERALIZATION           81.42%
=========================================================
```

Lalu benchmark dijalankan pada **seed yang belum pernah digunakan training**.

Ini sangat penting.

Karena:

```text
training success = 99%
```

belum tentu berarti:

```text
generalization = 99%
```

---

# 10. Pisahkan training environment dan evaluation environment

Saya sangat menyarankan ini.

```text
TRAIN SEEDS
1000–1999

VALIDATION SEEDS
2000–2999

TEST SEEDS
3000–3999
```

AI tidak boleh melihat test seeds selama training.

Baru kita bisa mengatakan:

> AI benar-benar belajar navigasi, bukan menghafal map.

Ini akan membuat project jauh lebih kredibel.

---

# 11. Visualisasikan apa yang dipelajari network

Ini cocok sekali dengan UI terminal yang sudah kamu buat.

Saat RUN tampilkan:

```text
Q0 FORWARD   +1.92
Q1 LEFT      +0.72
Q2 RIGHT     +2.81  <===
Q3 BRAKE     -0.14
Q4 REVERSE   -1.02
```

Kemudian:

```text
TARGET ANGLE   +0.38
DISTANCE       4.82
BEST ACTION    RIGHT
CONFIDENCE     0.74
```

Bahkan bisa dibuat:

```text
RAY 00 ██████████
RAY 01 █████████
RAY 02 ███████
RAY 03 ███
RAY 04 █
...
```

Sehingga programmer bisa melihat **mengapa AI memilih action**.

---

# 12. Training dashboard

UI COBOL/mainframe milikmu sebenarnya sangat cocok untuk telemetry.

Misalnya:

```text
TRAINING STATISTICS
-------------------

EPISODE        004812
STEP           0842193

EPSILON        .0712
LOSS           .0147
REWARD         41.28

SUCCESS        87.31%
COLLISION      09.82%

REPLAY         20000
Q-MEAN         1.82
Q-MAX          4.93

LEARNING RATE  .001000
GAMMA          .990000
TARGET SYNC    250
```

Dan buat **rolling statistics**, bukan hanya nilai episode terakhir.

---

# 13. Buat “AI replay”

Ini akan keren sekali.

Setelah training selesai:

```text
REPLAY 100
```

AI menjalankan 100 episode secara otomatis.

Kemudian:

```text
REPLAY EPISODE 042
```

memutar satu episode tertentu.

Bahkan bisa:

```text
REPLAY STEP 183
```

dan programmer bisa melihat:

```text
STATE
Q VALUES
RAYS
ACTION
REWARD
TARGET DISTANCE
```

persis sebelum action dilakukan.

---

# 14. Tambahkan mutation / procedural generation

Map jangan hanya random obstacle biasa.

Buat generator seperti:

```text
OPEN FIELD
CORRIDOR
U-SHAPE
DEAD END
MAZE
SNAKE
CROSS
ROOM
RANDOM CLUSTER
```

Kemudian training:

```text
MAP TYPE = RANDOM
```

dan AI harus bisa menghadapi semuanya.

Ini akan menguji apakah policy benar-benar general.

---

# 15. Buat reward sedapat mungkin tidak “menipu”

Ini sangat penting.

Reward shaping sekarang membantu AI belajar, tetapi terlalu banyak shaping dapat menghasilkan:

```text
AI pintar mendapatkan reward
```

vs

```text
AI belajar mengeksploitasi reward
```

Misalnya agent bisa berputar-putar dengan cara tertentu dan mendapatkan reward kecil terus-menerus.

Tambahkan diagnostics:

```text
REWARD COMPONENTS

PROGRESS       +0.032
HEADING        +0.002
CLEARANCE      +0.001
SUCCESS        +1.000
COLLISION      -1.000
```

Dengan demikian kita bisa melihat **mengapa total reward menjadi angka tertentu**.

---

# 16. Seed management

Setiap map sebaiknya memiliki:

```text
SEED 0000019281
```

dan command:

```text
ENV 19281
```

harus menghasilkan map yang sama.

Kemudian:

```text
RUN SEED 19281
```

bisa direproduksi.

Ini sangat penting untuk debugging AI.

---

# 17. AI league

Ini tahap yang sangat menarik.

Simpan beberapa checkpoint:

```text
MODEL-001
MODEL-002
MODEL-003
...
MODEL-050
```

kemudian:

```text
MODEL 001 vs 002
MODEL 002 vs 003
...
```

Benchmark semua model.

Hasil:

```text
MODEL       SUCCESS    AVG STEPS
001         21.3%      481
010         52.9%      321
020         74.1%      211
030         86.2%      164
040         91.8%      142
```

Sekarang kamu bisa benar-benar melihat **evolusi kemampuan AI**.

---

# 18. Pisahkan “AI brain” dari renderer

Ini juga penting secara arsitektur.

Idealnya:

```text
src/
├── ai/
│   ├── environment
│   ├── network
│   ├── replay
│   ├── trainer
│   ├── checkpoint
│   └── benchmark
│
├── workers/
│   └── training.worker.ts
│
└── components/
    ├── Renderer
    ├── Controls
    └── Benchmark
```

Renderer seharusnya hanya tahu:

```text
ACTION
STATE
WORLD
TELEMETRY
```

bukan detail algoritma training.

---

# 19. GPU ray tracer dan AI bisa benar-benar dipisahkan

Arsitektur akhirnya bisa menjadi:

```text
                ┌───────────────────┐
                │   Astro UI        │
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │ Runtime Controller │
                └───────┬─────┬─────┘
                        │     │
             ┌──────────▼─┐ ┌─▼──────────┐
             │ Rust/WASM  │ │  WebGPU    │
             │ AI Engine  │ │ RayTracer  │
             └────────────┘ └────────────┘
```

Jadi AI tidak tergantung pada rendering.

Ini akan membuat benchmark training jauh lebih cepat karena AI bisa menjalankan ribuan episode **tanpa menggambar canvas setiap frame**.

---

# 20. Target akhir yang menurut saya paling “perfect”

Saya akan membawa project ini ke kondisi:

```text
RAYTRC.AI
│
├── PROCEDURAL ENVIRONMENT
├── 32+ SENSOR RAYS
├── DOUBLE DUELING DQN
├── PRIORITIZED REPLAY
├── N-STEP RETURN
├── TARGET NETWORK
├── HUEBER LOSS
├── EXPERT BOOTSTRAP
├── CURRICULUM LEARNING
├── CHECKPOINT SYSTEM
├── REPRODUCIBLE SEEDS
├── TRAIN / VALIDATE / TEST SPLIT
├── AUTOMATED BENCHMARK
├── MODEL LEAGUE
├── Q-VALUE VISUALIZATION
├── REWARD BREAKDOWN
├── AI REPLAY
└── COBOL/ISPF MAINFRAME UI
```

Dan yang paling penting: **jangan langsung menambah 20 algoritma sekaligus**.

Urutan yang saya pilih untuk project-mu:

**Checkpoint lengkap → benchmark/evaluation → curriculum learning → prioritized replay → n-step DQN → dueling network → procedural map generator → model league.**

Dengan urutan itu, project-mu berubah dari **“DQN yang bisa bergerak”** menjadi **eksperimen AI navigation yang bisa diukur, direproduksi, dibandingkan, dan benar-benar menunjukkan peningkatan kemampuan dari training ke training**.
