/**
 * JARVIS Travel AI — Node.js Backend Server
 * Full AI engine implemented in Node.js (no Python required)
 * Includes: RAG pipeline, ranking algorithm, budget optimizer, chatbot
 */

const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const https = require("https");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

const JWT_SECRET = "bookme_jwt_secret_2024_secure_key";
const JWT_EXPIRES = "7d";

// ─── In-memory user store (replace with DB in production) ────────────────────
const users = []; // { id, name, email, passwordHash, createdAt, avatar }
let userIdCounter = 1;

function generateToken(user) {
  return jwt.sign(
    { id: user.id, email: user.email, name: user.name },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRES }
  );
}

function verifyToken(req, res, next) {
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith("Bearer ")) {
    return res.status(401).json({ error: "No token provided" });
  }
  try {
    req.user = jwt.verify(auth.slice(7), JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: "Invalid or expired token" });
  }
}

function safeUser(u) {
  return { id: u.id, name: u.name, email: u.email, createdAt: u.createdAt, avatar: u.avatar || null };
}

const app = express();
app.use(cors({ origin: ["http://localhost:3000", "http://127.0.0.1:3000"] }));
app.use(express.json());

// ─── Load travel data ─────────────────────────────────────────────────────────
const DATA_PATH = path.join(__dirname, "..", "ai-engine", "data", "travel_data.json");
const travelData = JSON.parse(fs.readFileSync(DATA_PATH, "utf8"));

// ─── Simple TF-IDF style embeddings (cosine similarity) ───────────────────────
function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(Boolean);
}

function buildTfIdf(docs) {
  const df = {};
  docs.forEach((doc) => {
    const tokens = new Set(tokenize(doc.text));
    tokens.forEach((t) => { df[t] = (df[t] || 0) + 1; });
  });
  return docs.map((doc) => {
    const tokens = tokenize(doc.text);
    const tf = {};
    tokens.forEach((t) => { tf[t] = (tf[t] || 0) + 1; });
    const vec = {};
    Object.entries(tf).forEach(([t, count]) => {
      const idf = Math.log((docs.length + 1) / ((df[t] || 0) + 1));
      vec[t] = (count / tokens.length) * idf;
    });
    return { ...doc, vec };
  });
}

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  keys.forEach((k) => {
    dot += (a[k] || 0) * (b[k] || 0);
    normA += (a[k] || 0) ** 2;
    normB += (b[k] || 0) ** 2;
  });
  return normA && normB ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
}

// Build document corpus
function buildDocuments() {
  const docs = [];
  travelData.destinations.forEach((dest) => {
    docs.push({
      id: dest.id, type: "destination", destination: dest.name,
      text: `${dest.name} ${dest.state} ${dest.description} ${dest.tags.join(" ")} ${dest.best_season.join(" ")}`,
      raw: dest,
    });
    dest.places?.forEach((p) => docs.push({
      id: p.id, type: "place", destination: dest.name,
      text: `${p.name} ${dest.name} ${p.description} ${p.category} ${p.tags.join(" ")}`,
      raw: p,
    }));
    dest.hotels?.forEach((h) => docs.push({
      id: h.id, type: "hotel", destination: dest.name,
      text: `${h.name} ${dest.name} ${h.description} ${h.tier} ${h.amenities.join(" ")}`,
      raw: h,
    }));
  });
  travelData.travel_tips?.forEach((tip, i) => docs.push({
    id: `tip_${i}`, type: "tip", destination: "general",
    text: `travel tip ${tip}`, raw: { tip },
  }));
  return docs;
}

const rawDocs = buildDocuments();
const ragDocs = buildTfIdf(rawDocs);

function ragSearch(query, topK = 8, destinationFilter = null) {
  const qTokens = tokenize(query);
  const qVec = {};
  qTokens.forEach((t) => { qVec[t] = (qVec[t] || 0) + 1; });

  let results = ragDocs.map((doc) => ({
    ...doc,
    score: cosineSimilarity(qVec, doc.vec),
  }));

  if (destinationFilter) {
    const df = destinationFilter.toLowerCase();
    results = results.filter(
      (r) => r.destination.toLowerCase() === df || r.text.toLowerCase().includes(df)
    );
  }

  return results.sort((a, b) => b.score - a.score).slice(0, topK);
}

// ─── Ranking Algorithm ────────────────────────────────────────────────────────
function rankPlaces(places, preferences, budgetPerDay) {
  const prefSet = new Set(preferences.map((p) => p.toLowerCase()));
  return places
    .map((p) => {
      const ratingScore = (p.rating || 3) / 5;
      const popScore = (p.popularity || 50) / 100;
      const placeTags = new Set((p.tags || []).map((t) => t.toLowerCase()));
      const overlap = [...placeTags].filter((t) => prefSet.has(t)).length;
      const prefScore = Math.min(overlap / Math.max(prefSet.size, 1), 1);
      const cost = p.cost || 0;
      const costEff = cost === 0 ? 1 : Math.max(0, 1 - cost / (budgetPerDay * 0.5));
      const total = 0.4 * ratingScore + 0.3 * popScore + 0.2 * prefScore + 0.1 * costEff;
      return { ...p, _score: parseFloat(total.toFixed(4)) };
    })
    .sort((a, b) => b._score - a._score);
}

// ─── Budget Optimizer ─────────────────────────────────────────────────────────
function optimizeBudget(total, days) {
  const ratios = { hotel: 0.35, food: 0.20, transport: 0.15, activities: 0.20, shopping: 0.07, emergency: 0.03 };
  const alloc = {};
  Object.entries(ratios).forEach(([k, r]) => { alloc[k] = Math.round(total * r); });
  return {
    total: alloc,
    per_day: {
      hotel_per_night: Math.round(alloc.hotel / days),
      food_per_day: Math.round(alloc.food / days),
      transport_per_day: Math.round(alloc.transport / days),
      activities_per_day: Math.round(alloc.activities / days),
    },
  };
}

// ─── Find destination ─────────────────────────────────────────────────────────
function findDestination(query) {
  const q = query.toLowerCase().trim();
  return travelData.destinations.find(
    (d) =>
      d.id.toLowerCase().includes(q) ||
      d.name.toLowerCase().includes(q) ||
      q.includes(d.name.toLowerCase()) ||
      q.includes(d.id.toLowerCase())
  );
}

function mapPreferences(prefs) {
  const mapping = {
    adventure: ["adventure", "trekking", "water_sports", "sports"],
    food: ["food", "culture", "local"],
    nightlife: ["nightlife", "party", "beach"],
    beaches: ["beach", "water_sports", "sunset"],
    history: ["history", "culture", "architecture"],
    nature: ["nature", "wildlife", "trekking", "peaceful"],
    shopping: ["shopping", "culture", "local"],
    romance: ["romance", "sunset", "peaceful", "luxury"],
  };
  const tags = [];
  prefs.forEach((p) => {
    const pl = p.toLowerCase();
    Object.entries(mapping).forEach(([k, v]) => {
      if (k.includes(pl) || pl.includes(k)) tags.push(...v);
    });
  });
  return tags.length ? [...new Set(tags)] : ["adventure", "culture", "food"];
}

// ─── Generate trip ─────────────────────────────────────────────────────────────
function generateTripData(destination, days, budget, preferences) {
  const dest = findDestination(destination);
  if (!dest) {
    return {
      error: true,
      message: `Destination '${destination}' not found. Try: Goa, Manali, Jaipur, Kerala, Udaipur, Shimla`,
      suggestions: ["Goa", "Manali", "Jaipur", "Kerala", "Udaipur", "Shimla"],
    };
  }

  const budgetPlan = optimizeBudget(budget, days);
  const mappedPrefs = mapPreferences(preferences);
  const ranked = rankPlaces(dest.places, mappedPrefs, budget / days);
  const affordable = ranked.filter((p) => p.cost <= budgetPlan.per_day.activities_per_day) || ranked;
  const hotels = [...(dest.hotels || [])].sort((a, b) => {
    const diff = Math.abs(a.price_per_night - budgetPlan.per_day.hotel_per_night) -
      Math.abs(b.price_per_night - budgetPlan.per_day.hotel_per_night);
    return diff;
  });
  const chosenHotel = hotels[0] || dest.hotels[0];

  // Build itinerary
  const itinerary = [];
  let totalActivitiesCost = 0;
  for (let day = 1; day <= days; day++) {
    const start = (day - 1) * 3;
    const dayPlaces = (affordable.length > start ? affordable.slice(start, start + 4) : affordable.slice(0, 4));
    let daySpent = 0, hoursUsed = 0;
    const acts = [];
    for (const p of dayPlaces) {
      if (hoursUsed + (p.duration_hours || 2) > 10) break;
      acts.push(p);
      daySpent += p.cost || 0;
      hoursUsed += p.duration_hours || 2;
    }
    totalActivitiesCost += daySpent;
    const food = dest.food?.[(day - 1) % (dest.food?.length || 1)];
    const transport = dest.transport?.[0];
    itinerary.push({
      day,
      title: `Day ${day} — ${acts.slice(0, 2).map((a) => a.name).join(", ") || dest.name}`,
      activities: acts,
      meals: food || null,
      transport: transport || null,
      estimated_cost: Math.round(daySpent + (food?.avg_cost_per_person || 200) * 2),
    });
  }

  // RAG tips
  const ragResults = ragSearch(`${destination} ${preferences.join(" ")} tips`, 5);
  const tips = ragResults.filter((r) => r.type === "tip").slice(0, 3).map((r) => r.raw.tip);

  const totalHotel = chosenHotel.price_per_night * days;
  const totalFood = budgetPlan.total.food;
  const totalTransport = budgetPlan.total.transport;
  const totalEst = totalHotel + totalFood + totalTransport + totalActivitiesCost;

  return {
    destination: dest.name,
    state: dest.state,
    days,
    total_budget: budget,
    description: dest.description,
    best_season: dest.best_season,
    hotel: chosenHotel,
    itinerary,
    budget_breakdown: {
      hotel: Math.round(totalHotel),
      food: Math.round(totalFood),
      transport: Math.round(totalTransport),
      activities: Math.round(totalActivitiesCost),
      shopping_emergency: Math.round(budgetPlan.total.shopping + budgetPlan.total.emergency),
      total_estimated: Math.round(totalEst),
      savings: Math.round(Math.max(0, budget - totalEst)),
    },
    tips,
    recommended_transport: (dest.transport || []).slice(0, 2),
    must_try_food: (dest.food || []).slice(0, 3),
    preference_tags: mappedPrefs,
    rag_context_used: ragResults.length,
  };
}

// ─── Chatbot engine ───────────────────────────────────────────────────────────
const sessions = {};
function getSession(id) {
  if (!sessions[id]) sessions[id] = { id, messages: [], ctx: { destination: null, days: null, budget: null, preferences: [] } };
  return sessions[id];
}

function extractInfo(text) {
  const lower = text.toLowerCase();
  const DESTS = ["goa", "manali", "jaipur", "kerala", "udaipur", "shimla", "rajasthan"];
  const dest = DESTS.find((d) => lower.includes(d));
  const dayMatch = lower.match(/(\d+)\s*(?:day|days|night)/);
  const budgetMatch = lower.match(/₹\s*([\d,]+)|(\d+)k\b|(\d[\d,]+)\s*(?:rs|rupees|inr)/);
  const prefs = ["adventure", "food", "nightlife", "beaches", "history", "nature", "shopping", "romance"].filter((p) => lower.includes(p));
  let budget = null;
  if (budgetMatch) {
    if (budgetMatch[2]) budget = parseInt(budgetMatch[2]) * 1000;
    else if (budgetMatch[1]) budget = parseInt(budgetMatch[1].replace(/,/g, ""));
    else if (budgetMatch[3]) budget = parseInt(budgetMatch[3].replace(/,/g, ""));
  }
  return {
    destination: dest ? dest.charAt(0).toUpperCase() + dest.slice(1) : null,
    days: dayMatch ? parseInt(dayMatch[1]) : null,
    budget,
    preferences: prefs,
  };
}

function chatResponse(sessionId, message) {
  const session = getSession(sessionId);
  const extracted = extractInfo(message);
  if (extracted.destination) session.ctx.destination = extracted.destination;
  if (extracted.days) session.ctx.days = extracted.days;
  if (extracted.budget) session.ctx.budget = extracted.budget;
  if (extracted.preferences.length) {
    session.ctx.preferences = [...new Set([...session.ctx.preferences, ...extracted.preferences])];
  }

  const lower = message.toLowerCase();
  const ctx = session.ctx;
  const isPlan = /plan|trip|travel|visit|itinerary|go to/.test(lower);
  const isGreet = /hello|hi|hey|jarvis|start|help/.test(lower);

  if (isGreet && !isPlan) {
    return {
      message: "🤖 **JARVIS Travel AI online.**\n\nI can plan personalized trips, find hotels, and optimize your budget!\n\nTry: *\"Plan a 3-day Goa trip under ₹15,000\"*",
      type: "greeting",
      suggestions: ["Plan a 3-day Goa trip under ₹15k", "Best hotels in Manali", "5-day Kerala tour for ₹25k", "Jaipur history + food tour"],
    };
  }

  if (ctx.destination && ctx.days && ctx.budget) {
    const trip = generateTripData(ctx.destination, ctx.days, ctx.budget, ctx.preferences);
    session.lastTrip = trip;
    if (trip.error) return { message: `❌ ${trip.message}`, type: "error", suggestions: trip.suggestions?.map((s) => `Plan a trip to ${s}`) };
    return {
      message: `🚀 **JARVIS generated your ${ctx.destination} itinerary!**\n\n📅 **${ctx.days} Days** | 💰 **₹${ctx.budget.toLocaleString()}**\n🏨 **${trip.hotel.name}** @ ₹${trip.hotel.price_per_night}/night\n\nFull plan ready in the dashboard!`,
      type: "trip_generated",
      trip,
      suggestions: [`What to eat in ${ctx.destination}?`, "Show cheapest hotels", "Add adventure activities", "Modify to 5 days"],
    };
  }

  // Ask for missing
  const missing = [];
  if (!ctx.destination) missing.push("Which destination? (Goa, Manali, Jaipur, Kerala, Udaipur, Shimla)");
  else if (!ctx.days) missing.push(`How many days in ${ctx.destination}?`);
  else if (!ctx.budget) missing.push("What's your total budget in ₹?");

  if (missing.length) return { message: missing[0], type: "clarification" };

  // RAG search for info
  const results = ragSearch(message, 5, ctx.destination);
  if (results.length) {
    const parts = results.slice(0, 3).map((r) => {
      if (r.type === "place") return `**${r.raw.name}** — ${r.raw.description?.slice(0, 80)}... (⭐${r.raw.rating}, ₹${r.raw.cost === 0 ? "Free" : r.raw.cost})`;
      if (r.type === "hotel") return `**${r.raw.name}** (${r.raw.tier}) — ₹${r.raw.price_per_night}/night, ⭐${r.raw.rating}`;
      if (r.type === "tip") return `💡 ${r.raw.tip}`;
      return `**${r.raw.name || r.destination}** — ${(r.raw.description || "").slice(0, 80)}`;
    });
    return { message: "Here's what I found:\n\n" + parts.map((p) => `• ${p}`).join("\n\n"), type: "info" };
  }

  return {
    message: "I can help with Goa, Manali, Jaipur, Kerala, Udaipur, and Shimla. Try: *\"Plan a 3-day Goa trip under ₹15,000\"*",
    type: "fallback",
    suggestions: ["Plan 3-day Goa trip under ₹15k", "Manali adventure tour", "Kerala backwaters tour"],
  };
}

// ─── Routes ───────────────────────────────────────────────────────────────────
app.get("/", (req, res) => res.json({ status: "JARVIS Travel AI online", engine: "Node.js", version: "1.0.0" }));
app.get("/health", (req, res) => res.json({ status: "healthy", docs: ragDocs.length }));

app.post("/generate", (req, res) => {
  const { destination, days, budget, preferences = [] } = req.body;
  if (!destination || !days || !budget) return res.status(400).json({ detail: "Missing required fields" });
  const result = generateTripData(destination, days, budget, preferences);
  res.json(result);
});

app.post("/chat", (req, res) => {
  const { session_id, message } = req.body;
  if (!message?.trim()) return res.status(400).json({ detail: "Empty message" });
  res.json(chatResponse(session_id || "default", message));
});

app.post("/search", (req, res) => {
  const { query, destination, top_k = 5 } = req.body;
  res.json({ query, results: ragSearch(query, top_k, destination), total: top_k });
});

app.get("/destinations", (req, res) => {
  res.json({
    destinations: travelData.destinations.map((d) => ({
      id: d.id, name: d.name, state: d.state, description: d.description,
      tags: d.tags, budget_per_day: d.budget_per_day, best_season: d.best_season,
    })),
  });
});

// ─── Weather proxy (avoids CORS issues) ──────────────────────────────────────
const DEST_COORDS = {
  goa: [15.2993, 74.124], manali: [32.2432, 77.1892], jaipur: [26.9124, 75.7873],
  kerala: [10.1632, 76.6413], udaipur: [24.5854, 73.7125], shimla: [31.1048, 77.1734],
};
app.get("/weather", (req, res) => {
  const dest = (req.query.destination || "").toLowerCase();
  const coords = DEST_COORDS[dest];
  if (!coords) return res.json(null);
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${coords[0]}&longitude=${coords[1]}&current_weather=true&forecast_days=1`;
  https.get(url, (apiRes) => {
    let data = "";
    apiRes.on("data", (chunk) => data += chunk);
    apiRes.on("end", () => {
      try {
        const json = JSON.parse(data);
        const cw = json.current_weather;
        const codes = { 0:"Clear sky",1:"Partly cloudy",2:"Partly cloudy",3:"Overcast",45:"Foggy",48:"Foggy",51:"Light drizzle",61:"Rain",71:"Snow",80:"Rain showers",95:"Thunderstorm" };
        res.json({ temperature: Math.round(cw.temperature), windspeed: Math.round(cw.windspeed), weathercode: cw.weathercode, description: codes[cw.weathercode] || "Clear sky" });
      } catch { res.json(null); }
    });
  }).on("error", () => res.json(null));
});

// ─── Auth Routes ──────────────────────────────────────────────────────────────

// Register
app.post("/auth/register", async (req, res) => {
  const { name, email, password } = req.body;
  if (!name?.trim() || !email?.trim() || !password?.trim()) {
    return res.status(400).json({ error: "Name, email, and password are required" });
  }
  if (password.length < 6) {
    return res.status(400).json({ error: "Password must be at least 6 characters" });
  }
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({ error: "Invalid email address" });
  }
  const existing = users.find((u) => u.email.toLowerCase() === email.toLowerCase());
  if (existing) {
    return res.status(409).json({ error: "An account with this email already exists" });
  }
  const passwordHash = await bcrypt.hash(password, 10);
  const user = {
    id: userIdCounter++,
    name: name.trim(),
    email: email.toLowerCase().trim(),
    passwordHash,
    createdAt: new Date().toISOString(),
    avatar: null,
  };
  users.push(user);
  const token = generateToken(user);
  console.log(`✅ New user registered: ${user.email}`);
  res.status(201).json({ token, user: safeUser(user), message: "Account created successfully" });
});

// Login
app.post("/auth/login", async (req, res) => {
  const { email, password } = req.body;
  if (!email?.trim() || !password?.trim()) {
    return res.status(400).json({ error: "Email and password are required" });
  }
  const user = users.find((u) => u.email.toLowerCase() === email.toLowerCase().trim());
  if (!user) {
    return res.status(401).json({ error: "No account found with this email" });
  }
  const valid = await bcrypt.compare(password, user.passwordHash);
  if (!valid) {
    return res.status(401).json({ error: "Incorrect password" });
  }
  const token = generateToken(user);
  console.log(`🔑 User logged in: ${user.email}`);
  res.json({ token, user: safeUser(user), message: "Welcome back!" });
});

// Get current user (protected)
app.get("/auth/me", verifyToken, (req, res) => {
  const user = users.find((u) => u.id === req.user.id);
  if (!user) return res.status(404).json({ error: "User not found" });
  res.json({ user: safeUser(user) });
});

// Update profile (protected)
app.patch("/auth/profile", verifyToken, async (req, res) => {
  const user = users.find((u) => u.id === req.user.id);
  if (!user) return res.status(404).json({ error: "User not found" });
  const { name, currentPassword, newPassword } = req.body;
  if (name) user.name = name.trim();
  if (currentPassword && newPassword) {
    const valid = await bcrypt.compare(currentPassword, user.passwordHash);
    if (!valid) return res.status(401).json({ error: "Current password is incorrect" });
    if (newPassword.length < 6) return res.status(400).json({ error: "New password must be at least 6 characters" });
    user.passwordHash = await bcrypt.hash(newPassword, 10);
  }
  const token = generateToken(user);
  res.json({ token, user: safeUser(user), message: "Profile updated" });
});

// Logout (client-side, just validate token)
app.post("/auth/logout", verifyToken, (req, res) => {
  res.json({ message: "Logged out successfully" });
});

const PORT = 8000;
app.listen(PORT, () => {
  console.log(`\n🏨 Bookme.com — Hotel Booking Server`);
  console.log(`   Running on http://localhost:${PORT}`);
  console.log(`   Auth: /auth/register, /auth/login, /auth/me`);
  console.log(`   RAG corpus: ${ragDocs.length} documents\n`);
});
