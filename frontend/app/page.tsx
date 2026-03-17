'use client';

import { useState, useRef, useEffect } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

interface Source {
  page: number;
  chapter: string;
  snippet: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

export default function Home() {
  const [mode, setMode] = useState<'chat' | 'notes' | 'summary'>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [notes, setNotes] = useState('');
  const [summary, setSummary] = useState('');
  const [summaryCount, setSummaryCount] = useState(0);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '20px';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const fetchSummary = async () => {
    setSummaryLoading(true);
    setSummary('');
    try {
      const res = await fetch(`${API_URL}/summary`);
      const data = await res.json();
      setSummaryCount(data.query_count);
      setSummary(data.summary || '');
    } catch {
      setSummary('Sorry, something went wrong fetching your summary.');
    } finally {
      setSummaryLoading(false);
    }
  };

  const handleModeChange = (newMode: 'chat' | 'notes' | 'summary') => {
    setMode(newMode);
    if (newMode === 'summary') fetchSummary();
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');

    if (mode === 'chat') {
      setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
      setLoading(true);

      try {
        const res = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: userMessage }),
        });
        const data = await res.json();
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: data.answer, sources: data.sources },
        ]);
      } catch (err) {
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' },
        ]);
      } finally {
        setLoading(false);
      }
    } else {
      // Notes mode
      setLoading(true);
      setNotes('');
      try {
        const res = await fetch(`${API_URL}/notes`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic: userMessage }),
        });
        const data = await res.json();
        setNotes(data.notes);
      } catch (err) {
        setNotes('Sorry, something went wrong generating notes. Please try again.');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestion = (text: string) => {
    setInput(text);
    setTimeout(() => textareaRef.current?.focus(), 0);
  };

  const chatSuggestions = [
    'What are the key concepts in Chapter 1?',
    'Explain pharmacology basics',
    'Summarize the microbiology section',
  ];

  const notesSuggestions = [
    'Pharmacology',
    'Microbiology',
    'Pathology',
  ];

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo-icon">📖</div>
          <div>
            <h1>STEP Study Assistant</h1>
            <span className="header-subtitle">Powered by RAG • 784 pages indexed</span>
          </div>
        </div>
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'chat' ? 'active' : ''}`}
            onClick={() => handleModeChange('chat')}
          >
            💬 Chat
          </button>
          <button
            className={`mode-btn ${mode === 'notes' ? 'active' : ''}`}
            onClick={() => handleModeChange('notes')}
          >
            📝 Notes
          </button>
          <button
            className={`mode-btn ${mode === 'summary' ? 'active' : ''}`}
            onClick={() => handleModeChange('summary')}
          >
            📊 Summary
          </button>
        </div>
      </header>

      {/* Chat Mode */}
      {mode === 'chat' && (
        <div className="messages-area">
          {messages.length === 0 && !loading && (
            <div className="welcome">
              <div className="welcome-icon">🧠</div>
              <h2>Ask anything about your textbook</h2>
              <p>
                I have your entire 800-page STEP textbook indexed. Ask me any question
                and I&apos;ll answer with precise page and chapter citations.
              </p>
              <div className="suggestions">
                {chatSuggestions.map((s, i) => (
                  <button key={i} className="suggestion-chip" onClick={() => handleSuggestion(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'assistant' ? '🤖' : '👤'}
              </div>
              <div className="message-content">
                {msg.content.split('\n').map((line, j) => (
                  <p key={j}>{line || '\u00A0'}</p>
                ))}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources-section">
                    <div className="sources-label">📚 Sources</div>
                    {msg.sources.map((src, j) => (
                      <span key={j} className="source-chip">
                        📄 Page {src.page} • {src.chapter}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-avatar">🤖</div>
              <div className="message-content">
                <div className="loading-dots">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      )}

      {/* Notes Mode */}
      {mode === 'notes' && (
        <div className="notes-content">
          {!notes && !loading && (
            <div className="welcome">
              <div className="welcome-icon">📝</div>
              <h2>Generate Study Notes</h2>
              <p>
                Enter a topic, chapter name, or concept and I&apos;ll generate
                detailed study notes with citations from your textbook.
              </p>
              <div className="suggestions">
                {notesSuggestions.map((s, i) => (
                  <button key={i} className="suggestion-chip" onClick={() => handleSuggestion(s)}>
                    📋 {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {loading && (
            <div className="notes-rendered">
              <div className="loading-dots">
                <span></span><span></span><span></span>
              </div>
              <p style={{ color: 'var(--text-muted)', marginTop: 8 }}>Generating notes...</p>
            </div>
          )}

          {notes && !loading && (
            <div className="notes-rendered" dangerouslySetInnerHTML={{
              __html: notes
                .replace(/### (.*)/g, '<h3>$1</h3>')
                .replace(/## (.*)/g, '<h2>$1</h2>')
                .replace(/# (.*)/g, '<h1>$1</h1>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/^- (.*)/gm, '<li>$1</li>')
                .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
                .replace(/\n/g, '<br/>')
            }} />
          )}
        </div>
      )}

      {/* Summary Mode */}
      {mode === 'summary' && (
        <div className="notes-content">
          {summaryLoading && (
            <div className="notes-rendered">
              <div className="loading-dots">
                <span></span><span></span><span></span>
              </div>
              <p style={{ color: 'var(--text-muted)', marginTop: 8 }}>Analyzing your session...</p>
            </div>
          )}

          {!summaryLoading && summaryCount < 2 && (
            <div className="welcome">
              <div className="welcome-icon">📊</div>
              <h2>No session data yet</h2>
              <p>
                Ask at least 2 questions in Chat mode and then come back here to see
                which concepts you kept returning to.
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 8 }}>
                {summaryCount === 1 ? '1 question asked so far today.' : '0 questions asked so far today.'}
              </p>
            </div>
          )}

          {!summaryLoading && summary && (
            <div className="notes-rendered">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Based on {summaryCount} question{summaryCount !== 1 ? 's' : ''} asked today
                </span>
                <button className="suggestion-chip" onClick={fetchSummary} style={{ fontSize: 12 }}>
                  🔄 Refresh
                </button>
              </div>
              <div dangerouslySetInnerHTML={{
                __html: summary
                  .replace(/### (.*)/g, '<h3>$1</h3>')
                  .replace(/## (.*)/g, '<h2>$1</h2>')
                  .replace(/# (.*)/g, '<h1>$1</h1>')
                  .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                  .replace(/\*(.*?)\*/g, '<em>$1</em>')
                  .replace(/`(.*?)`/g, '<code>$1</code>')
                  .replace(/^- (.*)/gm, '<li>$1</li>')
                  .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
                  .replace(/\n/g, '<br/>')
              }} />
            </div>
          )}
        </div>
      )}

      {/* Input Area — hidden in summary mode */}
      <div className="input-area" style={{ display: mode === 'summary' ? 'none' : undefined }}>
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              mode === 'chat'
                ? 'Ask a question about your textbook...'
                : 'Enter a topic to generate notes...'
            }
            rows={1}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={!input.trim() || loading}
          >
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}
