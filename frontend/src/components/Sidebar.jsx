import React, { useState } from 'react';
import { 
  Plus, 
  Pin, 
  History, 
  Trash2, 
  PanelLeftClose, 
  PanelLeftOpen, 
  Globe, 
  Settings, 
  Sun, 
  Moon, 
  MessageSquare, 
  Sparkles,
  ChevronRight,
  ShieldCheck
} from 'lucide-react';

export default function Sidebar({
  isOpen,
  setIsOpen,
  sessions = [],
  activeSessionId,
  onSelectSession,
  onNewChat,
  onTogglePin,
  onDeleteSession,
  darkMode,
  setDarkMode,
  onOpenSettings,
  onOpenCoords
}) {
  const pinnedSessions = sessions.filter(s => s.is_pinned === 1);
  const unpinnedSessions = sessions.filter(s => s.is_pinned !== 1);

  return (
    <aside
      className={`fixed top-0 left-0 bottom-0 z-40 bg-white/95 dark:bg-[#0c0d12]/95 border-r border-zinc-200 dark:border-zinc-800/80 backdrop-blur-xl flex flex-col transition-all duration-300 ease-out shadow-lg dark:shadow-2xl ${
        isOpen ? 'w-64' : 'w-16'
      }`}
    >
      {/* Top Header Logo & Collapse/Expand Toggle */}
      <div className="h-14 px-3.5 flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800/80 shrink-0">
        <div className="flex items-center space-x-2.5 overflow-hidden">
          <img src="/logo.png" alt="SatQuery Logo" className="h-7 w-7 shrink-0 object-contain" />
          {isOpen && (
            <div className="flex flex-col animate-fade-in">
              <span className="font-extrabold text-xs tracking-widest text-zinc-900 dark:text-white uppercase font-mono">
                SATQUERY
              </span>
              <span className="text-[9px] text-cyan-600 dark:text-cyan-400 font-mono font-medium">ISRO / SAC MEMORY</span>
            </div>
          )}
        </div>

        <button
          onClick={() => setIsOpen(!isOpen)}
          title={isOpen ? "Collapse Sidebar" : "Expand Sidebar"}
          className="impeccable-btn p-1.5 rounded-xl text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-800/60 transition-all cursor-pointer shrink-0"
        >
          {isOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-2.5 border-b border-zinc-200 dark:border-zinc-800/80 shrink-0">
        <button
          onClick={onNewChat}
          title="Start New Analysis Chat Session"
          className={`impeccable-btn w-full py-2.5 rounded-xl border border-cyan-500/40 dark:border-cyan-400/40 bg-cyan-500/5 dark:bg-cyan-500/10 text-cyan-700 dark:text-cyan-400 hover:bg-cyan-500/15 dark:hover:bg-cyan-400/20 font-bold text-xs flex items-center justify-center space-x-2 transition-all cursor-pointer shadow-xs ${
            isOpen ? 'px-3' : 'px-0'
          }`}
        >
          <Plus className="h-4 w-4 shrink-0 text-cyan-600 dark:text-cyan-400" />
          {isOpen && <span className="truncate">New Chat</span>}
        </button>
      </div>

      {/* Sidebar Navigation Items List — Only rendered when sidebar is open */}
      {isOpen ? (
        <div className="flex-1 overflow-y-auto px-2 py-3 space-y-4">
          
          {/* Pinned Chats Section */}
          {pinnedSessions.length > 0 && (
            <div className="space-y-1">
              <div className="px-2 pb-1 text-[10px] font-mono font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider flex items-center space-x-1.5">
                <Pin className="h-3 w-3 fill-amber-600 dark:fill-amber-400 text-amber-600 dark:text-amber-400" />
                <span>Pinned Chats</span>
              </div>

              {pinnedSessions.map((session) => (
                <div
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  className={`group relative flex items-center p-2 rounded-xl text-xs transition-all cursor-pointer ${
                    activeSessionId === session.id
                      ? 'bg-cyan-500/10 dark:bg-cyan-950/40 text-cyan-700 dark:text-cyan-400 font-semibold border border-cyan-500/30'
                      : 'text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/50'
                  }`}
                  title={session.title}
                >
                  <Pin className="h-4 w-4 shrink-0 text-amber-500 fill-amber-500" />

                  <div className="ml-2.5 flex-1 overflow-hidden pr-6">
                    <div className="truncate font-medium text-xs text-zinc-900 dark:text-white">{session.title}</div>
                    <div className="truncate text-[10px] text-zinc-500 dark:text-zinc-400 font-normal">{session.subtitle}</div>
                  </div>

                  {/* Quick Action Icons on Hover */}
                  <div className="absolute right-1.5 flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => onTogglePin(session.id, e)}
                      title="Unpin Chat"
                      className="p-1 rounded-lg text-amber-500 hover:bg-amber-100 dark:hover:bg-amber-500/20 cursor-pointer"
                    >
                      <Pin className="h-3 w-3 fill-amber-500" />
                    </button>
                    <button
                      onClick={(e) => onDeleteSession(session.id, e)}
                      title="Delete Session"
                      className="p-1 rounded-lg text-zinc-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 cursor-pointer"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Recent History Section */}
          <div className="space-y-1">
            <div className="px-2 pb-1 text-[10px] font-mono font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider flex items-center space-x-1.5">
              <History className="h-3 w-3 text-zinc-500 dark:text-zinc-400" />
              <span>History</span>
            </div>

            {unpinnedSessions.length === 0 && pinnedSessions.length === 0 ? (
              <div className="px-2 py-4 text-center text-xs text-zinc-400 font-mono">
                No past sessions saved.
              </div>
            ) : (
              unpinnedSessions.map((session) => (
                <div
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  className={`group relative flex items-center p-2 rounded-xl text-xs transition-all cursor-pointer ${
                    activeSessionId === session.id
                      ? 'bg-cyan-500/10 dark:bg-cyan-950/40 text-cyan-700 dark:text-cyan-400 font-semibold border border-cyan-500/30'
                      : 'text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/50'
                  }`}
                  title={session.title}
                >
                  <MessageSquare className="h-4 w-4 shrink-0 text-zinc-400 dark:text-zinc-500 group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors" />

                  <div className="ml-2.5 flex-1 overflow-hidden pr-12">
                    <div className="truncate font-medium text-xs text-zinc-900 dark:text-white">{session.title}</div>
                    <div className="truncate text-[10px] text-zinc-500 dark:text-zinc-400 font-normal">{session.subtitle}</div>
                  </div>

                  {/* Quick Action Icons on Hover */}
                  <div className="absolute right-1.5 flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => onTogglePin(session.id, e)}
                      title="Pin Chat to Top"
                      className="p-1 rounded-lg text-zinc-400 hover:text-amber-500 hover:bg-zinc-200/80 dark:hover:bg-zinc-700/50 cursor-pointer"
                    >
                      <Pin className="h-3 w-3" />
                    </button>
                    <button
                      onClick={(e) => onDeleteSession(session.id, e)}
                      title="Delete Session"
                      className="p-1 rounded-lg text-zinc-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 cursor-pointer"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1" />
      )}

      {/* Bottom Footer Actions */}
      <div className="p-2 border-t border-zinc-200 dark:border-zinc-800/80 space-y-1 shrink-0 bg-zinc-50/50 dark:bg-transparent">
        
        {/* Geospatial Coords Modal Launcher */}
        <button
          onClick={onOpenCoords}
          title="Geospatial Coordinates & Satellite Fetcher"
          className={`impeccable-btn w-full p-2 rounded-xl text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200/60 dark:hover:bg-zinc-800/60 flex items-center space-x-2.5 transition-all cursor-pointer ${
            isOpen ? '' : 'justify-center'
          }`}
        >
          <Globe className="h-4 w-4 text-cyan-600 dark:text-cyan-400 shrink-0" />
          {isOpen && <span className="truncate">Coords & Radius</span>}
        </button>

        {/* System Settings Gear */}
        <button
          onClick={onOpenSettings}
          title="System Settings & Telemetry"
          className={`impeccable-btn w-full p-2 rounded-xl text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200/60 dark:hover:bg-zinc-800/60 flex items-center space-x-2.5 transition-all cursor-pointer ${
            isOpen ? '' : 'justify-center'
          }`}
        >
          <Settings className="h-4 w-4 text-zinc-500 dark:text-zinc-400 shrink-0" />
          {isOpen && <span className="truncate">Settings & Hardware</span>}
        </button>

        {/* Dark/Light Mode Switch */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          className={`impeccable-btn w-full p-2 rounded-xl text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200/60 dark:hover:bg-zinc-800/60 flex items-center space-x-2.5 transition-all cursor-pointer ${
            isOpen ? '' : 'justify-center'
          }`}
        >
          {darkMode ? (
            <Moon className="h-4 w-4 text-cyan-400 shrink-0" />
          ) : (
            <Sun className="h-4 w-4 text-amber-500 shrink-0" />
          )}
          {isOpen && <span className="truncate font-medium">{darkMode ? 'Dark Theme' : 'Light Theme'}</span>}
        </button>
      </div>
    </aside>
  );
}
