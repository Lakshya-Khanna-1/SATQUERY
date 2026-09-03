import React, { useState, useEffect, useRef } from 'react';
import { Globe, MapPin, Map, Navigation, Check, X, Layers, RefreshCw, Compass } from 'lucide-react';

const PRESET_LOCATIONS = [
  { name: "ISRO Space Applications Centre (SAC), Ahmedabad", lat: 23.0225, lon: 72.5714 },
  { name: "Cartosat-2S Urban Test Corridor, Delhi", lat: 28.6139, lon: 77.2090 },
  { name: "LEVIR-CC Bi-Temporal Change Region, Texas", lat: 31.9686, lon: -99.9018 },
  { name: "Bengaluru Technology Corridor", lat: 12.9716, lon: 77.5946 }
];

export default function CoordsModal({ isOpen, onClose, onApplyCoords, activeTab = "n8n" }) {
  const [activeSubTab, setActiveSubTab] = useState("map"); // 'map' or 'manual'
  const [mapStyle, setMapStyle] = useState("satellite"); // 'satellite', 'dark', 'street'
  const [lat, setLat] = useState(23.0225);
  const [lon, setLon] = useState(72.5714);
  const [radiusKm, setRadiusKm] = useState(5);
  const [customRadius, setCustomRadius] = useState("");
  const [yearsAgo, setYearsAgo] = useState(5);
  const [fetching, setFetching] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);
  const circleRef = useRef(null);
  const tileLayerRef = useRef(null);

  // Wheel scroller event handler to adjust radius (500m to 10km)
  const handleRadiusWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.2 : -0.2;
    setRadiusKm((prev) => {
      const updated = Math.min(10, Math.max(0.5, Math.round((prev + delta) * 10) / 10));
      return updated;
    });
    setCustomRadius("");
  };

  // Initialize or re-render Leaflet Map cleanly on modal open/tab switch
  useEffect(() => {
    if (!isOpen || activeSubTab !== "map" || !mapContainerRef.current) return;

    let timer = setTimeout(() => {
      if (!mapContainerRef.current) return;

      // Safely destroy existing map instance if container re-mounted
      if (mapInstanceRef.current) {
        try {
          mapInstanceRef.current.remove();
        } catch (e) {
          console.warn("Clean map removal:", e);
        }
        mapInstanceRef.current = null;
      }

      if (window.L) {
        const L = window.L;
        const map = L.map(mapContainerRef.current, {
          center: [lat, lon],
          zoom: 12,
          zoomControl: true
        });
        mapInstanceRef.current = map;

        // Add default tile layer
        const getTileUrl = (style) => {
          if (style === "satellite") {
            return "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
          }
          return "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
        };

        const tileLayer = L.tileLayer(getTileUrl(mapStyle), {
          maxZoom: 18,
          attribution: '© OpenStreetMap / Esri'
        }).addTo(map);
        tileLayerRef.current = tileLayer;

        // Drop initial marker & circle
        const marker = L.marker([lat, lon], { draggable: true }).addTo(map);
        markerRef.current = marker;

        const circle = L.circle([lat, lon], {
          radius: (customRadius ? parseFloat(customRadius) || 5 : radiusKm) * 1000,
          color: '#06b6d4',
          fillColor: '#06b6d4',
          fillOpacity: 0.15,
          weight: 2
        }).addTo(map);
        circleRef.current = circle;

        // Map click handler to update position
        map.on('click', (e) => {
          const { lat: newLat, lng: newLng } = e.latlng;
          setLat(roundCoord(newLat));
          setLon(roundCoord(newLng));
        });

        marker.on('dragend', () => {
          const position = marker.getLatLng();
          setLat(roundCoord(position.lat));
          setLon(roundCoord(position.lng));
        });

        setTimeout(() => map.invalidateSize(), 150);
      }
    }, 50);

    return () => {
      clearTimeout(timer);
      if (mapInstanceRef.current) {
        try {
          mapInstanceRef.current.remove();
        } catch (e) {}
        mapInstanceRef.current = null;
      }
    };
  }, [isOpen, activeSubTab]);

  // Update map layer style
  useEffect(() => {
    if (mapInstanceRef.current && window.L && tileLayerRef.current) {
      const L = window.L;
      mapInstanceRef.current.removeLayer(tileLayerRef.current);
      
      let newUrl = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
      if (mapStyle === "street") {
        newUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
      }

      tileLayerRef.current = L.tileLayer(newUrl, { maxZoom: 18 }).addTo(mapInstanceRef.current);
    }
  }, [mapStyle]);

  // Update marker and circle position when lat/lon/radius change
  useEffect(() => {
    const currentRadius = customRadius ? parseFloat(customRadius) || 5 : radiusKm;
    if (mapInstanceRef.current && markerRef.current && circleRef.current) {
      markerRef.current.setLatLng([lat, lon]);
      circleRef.current.setLatLng([lat, lon]);
      circleRef.current.setRadius(currentRadius * 1000);
      mapInstanceRef.current.panTo([lat, lon]);
    }
  }, [lat, lon, radiusKm, customRadius]);

  const roundCoord = (val) => Math.round(val * 10000) / 10000;

  if (!isOpen) return null;

  const effectiveRadius = customRadius ? parseFloat(customRadius) || 5 : radiusKm;

  // Client-side high-resolution satellite scene generator (Exact circumscribed square around circle)
  const generateRealSatellitePair = async (targetLat, targetLon, radiusKmVal) => {
    const radiusM = (parseFloat(radiusKmVal) || 5) * 1000.0;
    const C = 40075016.686;
    const phi = (targetLat * Math.PI) / 180.0;
    const cosPhi = Math.cos(phi);
    
    // Optimal zoom so the circumscribed square has between 512 and 1024 px
    const targetPx = 650;
    let zoom = Math.round(Math.log2((targetPx * C * cosPhi) / (2.0 * radiusM * 256.0)));
    zoom = Math.max(11, Math.min(18, zoom));
    
    const n = Math.pow(2, zoom);
    const xFrac = ((targetLon + 180.0) / 360.0) * n;
    const yFrac = (1.0 - Math.log(Math.tan(phi) + 1.0 / Math.cos(phi)) / Math.PI) / 2.0 * n;
    
    const mPerPx = (C * cosPhi) / (256.0 * n);
    const H = radiusM / mPerPx; // Half-side of the circumscribed square in pixels
    
    const cx = xFrac * 256.0;
    const cy = yFrac * 256.0;
    
    const minX = cx - H;
    const maxX = cx + H;
    const minY = cy - H;
    const maxY = cy + H;
    
    const tMinX = Math.floor(minX / 256.0);
    const tMaxX = Math.floor(maxX / 256.0);
    const tMinY = Math.floor(minY / 256.0);
    const tMaxY = Math.floor(maxY / 256.0);
    
    const cols = tMaxX - tMinX + 1;
    const rows = tMaxY - tMinY + 1;
    
    const canvasT1 = document.createElement("canvas");
    canvasT1.width = cols * 256;
    canvasT1.height = rows * 256;
    const ctx1 = canvasT1.getContext("2d");
    
    const canvasT2 = document.createElement("canvas");
    canvasT2.width = cols * 256;
    canvasT2.height = rows * 256;
    const ctx2 = canvasT2.getContext("2d");
    
    const parsedYears = parseInt(yearsAgo) || 5;
    const currentYear = new Date().getFullYear();
    const historicalYear = Math.max(2017, Math.min(2023, currentYear - parsedYears));
    
    const tileCoords = [];
    for (let ty = tMinY; ty <= tMaxY; ty++) {
      for (let tx = tMinX; tx <= tMaxX; tx++) {
        tileCoords.push({
          tx,
          ty,
          px: (tx - tMinX) * 256,
          py: (ty - tMinY) * 256
        });
      }
    }
    
    const loadTile = (url, fallbackColor) => new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = () => {
        const c = document.createElement("canvas");
        c.width = 256; c.height = 256;
        const ctx = c.getContext("2d");
        ctx.fillStyle = fallbackColor || "#1e293b";
        ctx.fillRect(0, 0, 256, 256);
        resolve(c);
      };
      img.src = url;
    });

    const t1Promises = tileCoords.map(t => {
      const url = zoom <= 14
        ? `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-${historicalYear}_3857/default/GoogleMapsCompatible/${zoom}/${t.ty}/${t.tx}.jpg`
        : `https://mt1.google.com/vt/lyrs=s&x=${t.tx}&y=${t.ty}&z=${zoom}`;
      return loadTile(url, "#2d3748").then(tile => {
        ctx1.drawImage(tile, t.px, t.py, 256, 256);
      });
    });

    const t2Promises = tileCoords.map(t => {
      const url = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${zoom}/${t.ty}/${t.tx}`;
      return loadTile(url, "#1a202c").then(tile => {
        ctx2.drawImage(tile, t.px, t.py, 256, 256);
      });
    });

    await Promise.all([...t1Promises, ...t2Promises]);
    
    // Crop the exact circumscribed square out of the stitched canvas
    const cropX = Math.round(minX - tMinX * 256.0);
    const cropY = Math.round(minY - tMinY * 256.0);
    const cropW = Math.round(maxX - minX);
    const cropH = Math.round(maxY - minY);
    
    // Final 512x512 high-resolution square canvas
    const finalCanvas1 = document.createElement("canvas");
    finalCanvas1.width = 512;
    finalCanvas1.height = 512;
    const finalCtx1 = finalCanvas1.getContext("2d");
    finalCtx1.imageSmoothingQuality = "high";
    finalCtx1.drawImage(canvasT1, cropX, cropY, cropW, cropH, 0, 0, 512, 512);

    const finalCanvas2 = document.createElement("canvas");
    finalCanvas2.width = 512;
    finalCanvas2.height = 512;
    const finalCtx2 = finalCanvas2.getContext("2d");
    finalCtx2.imageSmoothingQuality = "high";
    finalCtx2.drawImage(canvasT2, cropX, cropY, cropW, cropH, 0, 0, 512, 512);

    return {
      t1_url: finalCanvas1.toDataURL("image/png"),
      t2_url: finalCanvas2.toDataURL("image/png")
    };
  };

  const handleApply = async () => {
    setFetching(true);
    setErrorMsg("");

    const targetUrl = activeTab === "n8n"
      ? "http://127.0.0.1:5678/fetch-coords-satellite"
      : "http://127.0.0.1:8000/fetch-coords-satellite";

    try {
      // First try backend satellite fetcher
      const response = await fetch(targetUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: parseFloat(lat),
          lon: parseFloat(lon),
          radius_km: effectiveRadius,
          years_ago: parseInt(yearsAgo)
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.time2_base64 && data.time2_base64.startsWith("data:image")) {
          onApplyCoords({
            image_url: data.time2_base64,
            time1_url: data.time1_base64,
            lat: data.lat,
            lon: data.lon,
            radius_km: data.radius_km,
            years_ago: data.years_ago
          });
          onClose();
          return;
        }
      }
      throw new Error("Backend tile fetcher fallback");

    } catch (err) {
      console.info("Using high-speed client-side real satellite tile engine:", err);
      // Client-side real satellite tile generator
      const realTiles = await generateRealSatellitePair(parseFloat(lat), parseFloat(lon), effectiveRadius);
      
      onApplyCoords({
        image_url: realTiles.t2_url,
        time1_url: realTiles.t1_url,
        lat: roundCoord(lat),
        lon: roundCoord(lon),
        radius_km: effectiveRadius,
        years_ago: yearsAgo
      });
      onClose();
    } finally {
      setFetching(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs animate-fade-in">
      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl max-w-2xl w-full p-5 shadow-2xl space-y-4 max-h-[90vh] flex flex-col">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-100 dark:border-zinc-800 pb-3">
          <div className="flex items-center space-x-2">
            <Globe className="h-5 w-5 text-cyan-500" />
            <div>
              <h3 className="font-bold text-sm text-zinc-900 dark:text-white">Geospatial Coordinate & Radius Selector</h3>
              <p className="text-[11px] text-zinc-500">Retrieve bi-temporal satellite data for any geographic region</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 p-1 rounded-lg cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Tab Switcher: Choose on Map vs Enter Manually */}
        <div className="flex items-center justify-between bg-zinc-100 dark:bg-zinc-800/60 p-1 rounded-xl text-xs font-medium">
          <div className="flex space-x-1">
            <button
              onClick={() => setActiveSubTab("map")}
              className={`impeccable-btn px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center space-x-1.5 ${
                activeSubTab === "map"
                  ? "bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs font-semibold"
                  : "text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
              }`}
            >
              <Map className="h-3.5 w-3.5 text-cyan-500" />
              <span>Choose on Map</span>
            </button>
            <button
              onClick={() => setActiveSubTab("manual")}
              className={`impeccable-btn px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center space-x-1.5 ${
                activeSubTab === "manual"
                  ? "bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs font-semibold"
                  : "text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
              }`}
            >
              <MapPin className="h-3.5 w-3.5 text-emerald-500" />
              <span>Enter Manually</span>
            </button>
          </div>

          {/* Map Layer Switcher (Satellite, Street) */}
          {activeSubTab === "map" && (
            <div className="flex items-center space-x-1 pr-1 text-[11px]">
              <button
                onClick={() => setMapStyle("satellite")}
                className={`impeccable-btn px-2.5 py-1 rounded-md font-medium transition-all ${mapStyle === "satellite" ? "bg-cyan-500 text-white font-bold shadow-xs" : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"}`}
              >
                Real
              </button>
              <button
                onClick={() => setMapStyle("street")}
                className={`impeccable-btn px-2.5 py-1 rounded-md font-medium transition-all ${mapStyle === "street" ? "bg-cyan-500 text-white font-bold shadow-xs" : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"}`}
              >
                Street
              </button>
            </div>
          )}
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto space-y-4">
          
          {/* MAP VIEW WITH SIDE ELEMENT SCROLLER */}
          {activeSubTab === "map" && (
            <div className="space-y-2">
              <div className="flex flex-col sm:flex-row gap-3 items-stretch">
                {/* Main Interactive Map */}
                <div className="relative h-64 sm:h-72 flex-1 rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 shadow-inner">
                  <div ref={mapContainerRef} className="h-full w-full z-10" />
                  
                  {/* Coordinates pill on map */}
                  <div className="absolute bottom-2 left-2 z-[400] bg-black/75 backdrop-blur-xs text-white px-2.5 py-1 rounded-lg text-[10px] font-mono pointer-events-none flex items-center space-x-1.5 shadow-md border border-white/10">
                    <MapPin className="h-3 w-3 text-cyan-400" />
                    <span>{lat}° N, {lon}° E</span>
                  </div>
                </div>

                {/* Element Scroller (Side of the Map Only: 500m to 10km) */}
                <div 
                  onWheel={handleRadiusWheel}
                  className="w-full sm:w-28 flex sm:flex-col justify-between items-center bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200 dark:border-zinc-700/60 rounded-xl p-2.5 sm:p-3 shadow-xs select-none"
                  title="Scroll wheel or drag to adjust radius (500m to 10km)"
                >
                  <div className="text-center w-full">
                    <span className="block text-[9px] uppercase tracking-wider font-bold text-zinc-400">Radius</span>
                    <span className="inline-block mt-0.5 px-2 py-0.5 rounded-md bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 font-mono font-bold text-xs border border-cyan-500/20">
                      {radiusKm < 1 ? `${Math.round(radiusKm * 1000)}m` : `${radiusKm.toFixed(1)}km`}
                    </span>
                  </div>

                  {/* Vertical Range Scroller */}
                  <div className="flex items-center space-x-2 sm:space-x-0 sm:flex-col justify-center my-1 sm:my-2">
                    <span className="text-[9px] font-mono text-zinc-400 sm:mb-1">10km</span>
                    <input
                      type="range"
                      min="0.5"
                      max="10.0"
                      step="0.1"
                      value={radiusKm}
                      onChange={(e) => {
                        setRadiusKm(parseFloat(e.target.value));
                        setCustomRadius("");
                      }}
                      style={{ writingMode: 'vertical-lr', direction: 'rtl' }}
                      className="h-24 sm:h-28 w-2 accent-cyan-500 cursor-pointer rounded-lg bg-zinc-200 dark:bg-zinc-700"
                    />
                    <span className="text-[9px] font-mono text-zinc-400 sm:mt-1">500m</span>
                  </div>

                  {/* Quick Scale Badges */}
                  <div className="grid grid-cols-2 gap-1 w-full text-[10px] font-mono">
                    {[0.5, 1, 5, 10].map((r) => (
                      <button
                        key={r}
                        type="button"
                        onClick={() => {
                          setRadiusKm(r);
                          setCustomRadius("");
                        }}
                        className={`px-1 py-1 rounded text-center transition-all cursor-pointer ${
                          Math.abs(radiusKm - r) < 0.05
                            ? 'bg-cyan-500 text-white font-bold shadow-xs'
                            : 'bg-zinc-200/60 dark:bg-zinc-700/60 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-300 dark:hover:bg-zinc-600'
                        }`}
                      >
                        {r < 1 ? '500m' : `${r}km`}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="text-[11px] text-zinc-500 flex items-center justify-between px-1">
                <span>Click/drag map marker • Scroll or drag side bar to change radius</span>
                <span className="font-mono text-cyan-600 dark:text-cyan-400">
                  Circle: {radiusKm < 1 ? `${Math.round(radiusKm * 1000)}m` : `${radiusKm.toFixed(1)}km`}
                </span>
              </div>
            </div>
          )}

          {/* MANUAL INPUT VIEW */}
          {activeSubTab === "manual" && (
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-zinc-500 dark:text-zinc-400 font-medium mb-1">Latitude (°N)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={lat}
                  onChange={(e) => setLat(parseFloat(e.target.value) || 0)}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-zinc-500 dark:text-zinc-400 font-medium mb-1">Longitude (°E)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={lon}
                  onChange={(e) => setLon(parseFloat(e.target.value) || 0)}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white focus:outline-none"
                />
              </div>
            </div>
          )}

          {/* Preset Locations Quick Chips */}
          <div>
            <label className="block text-xs text-zinc-500 dark:text-zinc-400 font-medium mb-1.5">Preset Target Locations</label>
            <div className="flex flex-wrap gap-1.5">
              {PRESET_LOCATIONS.map((loc, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setLat(loc.lat);
                    setLon(loc.lon);
                  }}
                  className="text-[11px] px-2.5 py-1 rounded-lg bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 transition-colors cursor-pointer"
                >
                  {loc.name}
                </button>
              ))}
            </div>
          </div>

          {/* Radius Selector for Manual tab */}
          {activeSubTab === "manual" && (
            <div className="space-y-1.5 text-xs">
              <label className="block text-zinc-500 dark:text-zinc-400 font-medium">Search Radius (km)</label>
              <div className="flex items-center space-x-2">
                {[0.5, 1, 5, 10].map((r) => (
                  <button
                    key={r}
                    onClick={() => {
                      setRadiusKm(r);
                      setCustomRadius("");
                    }}
                    className={`px-3 py-1.5 rounded-xl border text-xs font-mono transition-all cursor-pointer ${
                      radiusKm === r && !customRadius
                        ? "bg-cyan-500 text-white font-bold border-cyan-500"
                        : "border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-400"
                    }`}
                  >
                    {r < 1 ? '500m' : `${r} km`}
                  </button>
                ))}
                <input
                  type="number"
                  placeholder="Custom radius..."
                  value={customRadius}
                  onChange={(e) => setCustomRadius(e.target.value)}
                  className="w-28 bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2.5 py-1.5 text-xs text-zinc-900 dark:text-white focus:outline-none"
                />
              </div>
            </div>
          )}

          {/* Historical Timeframe Selector */}
          <div className="space-y-1.5 text-xs">
            <label className="block text-zinc-500 dark:text-zinc-400 font-medium">Historical Comparison Baseline</label>
            <div className="flex items-center space-x-2">
              {[1, 3, 5, 10].map((y) => (
                <button
                  key={y}
                  onClick={() => setYearsAgo(y)}
                  className={`px-3 py-1.5 rounded-xl border text-xs font-mono transition-all cursor-pointer ${
                    yearsAgo === y
                      ? "bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 font-bold border-transparent"
                      : "border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-400"
                  }`}
                >
                  {y} {y === 1 ? 'Year' : 'Years'} Ago
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between pt-3 border-t border-zinc-100 dark:border-zinc-800">
          <div className="text-[11px] font-mono text-zinc-500">
            Selected: <span className="font-bold text-zinc-900 dark:text-white">{effectiveRadius} km</span> radius around <span className="font-bold text-zinc-900 dark:text-white">{lat}°N, {lon}°E</span> ({yearsAgo} yrs ago baseline)
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-xl border border-zinc-200 dark:border-zinc-800 text-xs text-zinc-600 dark:text-zinc-400 cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={handleApply}
              disabled={fetching}
              className="px-4 py-1.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium cursor-pointer shadow-xs flex items-center space-x-1.5 disabled:opacity-50"
            >
              {fetching ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                  <span>Fetching Satellites...</span>
                </>
              ) : (
                <>
                  <Check className="h-3.5 w-3.5" />
                  <span>Apply Target Region</span>
                </>
              )}
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
