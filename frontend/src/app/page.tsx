'use client';

import { useState, useEffect, useRef } from 'react';
import { MovieCard, Movie } from '@/components/MovieCard';
import { Search, X, Users, AlertCircle, Loader2, Film, Sparkles, Star, Sliders } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

interface SearchResult {
  movie_id: number;
  title: string;
  genres: string;
  poster_url: string;
  overview?: string;
  score?: number;
}

export default function Home() {
  const [userId, setUserId] = useState<number>(1);
  const [mode, setMode] = useState<'hybrid' | 'collaborative' | 'content' | 'compare'>('hybrid');
  const [movies, setMovies] = useState<Movie[]>([]);
  const [comparisonData, setComparisonData] = useState<{ [key: string]: Movie[] } | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [apiError, setApiError] = useState<string | null>(null);

  // MMR Diversity Parameter (1.0 = Max Match, 0.0 = Max Diversity)
  const [diversityLambda, setDiversityLambda] = useState<number>(0.65);

  // Search & Seed Discovery State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [activeSeed, setActiveSeed] = useState<SearchResult | null>(null);
  const searchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!activeSeed) {
      fetchRecommendations();
    }
  }, [userId, mode, activeSeed]);

  // Click outside to dismiss search results
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setSearchResults([]);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Live search debounced
  useEffect(() => {
    const query = searchQuery.trim();
    if (query.length < 2) {
      setSearchResults([]);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    const controller = new AbortController();

    const timer = setTimeout(async () => {
      try {
        const res = await fetch(
          `${API_BASE}/search?query=${encodeURIComponent(query)}&limit=6`,
          { signal: controller.signal }
        );
        if (!res.ok) {
          setSearchResults([]);
          return;
        }
        const data = await res.json();
        setSearchResults(Array.isArray(data) ? data : []);
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          setSearchResults([]);
        }
      } finally {
        setIsSearching(false);
      }
    }, 250);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [searchQuery]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setApiError(null);
    try {
      if (mode === 'compare') {
        const res = await fetch(`${API_BASE}/compare?user_id=${userId}`);
        if (!res.ok) throw new Error(`Status ${res.status}`);
        const data = await res.json();
        setComparisonData(data);
      } else {
        const res = await fetch(`${API_BASE}/recommend/${userId}?mode=${mode}`);
        if (!res.ok) throw new Error(`Status ${res.status}`);
        const data = await res.json();
        setMovies(data);
        setComparisonData(null);
      }
    } catch (err) {
      console.error('API connection error:', err);
      setApiError('Unable to connect to backend engine. Verify FastAPI is running at http://127.0.0.1:8000.');
    } finally {
      setLoading(false);
    }
  };

  const exploreSimilar = async (film: SearchResult, lambdaVal = diversityLambda) => {
    setActiveSeed(film);
    setSearchResults([]);
    setSearchQuery('');
    setLoading(true);
    setApiError(null);
    try {
      const res = await fetch(`${API_BASE}/movie/${film.movie_id}/similar?diversity=${lambdaVal}`);
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      setMovies(data);
      setComparisonData(null);
    } catch (err) {
      console.error('Vector retrieval error:', err);
      setApiError('Unable to find similar titles at this time.');
    } finally {
      setLoading(false);
    }
  };

  const resetToProfile = () => {
    setActiveSeed(null);
  };

  const seedYearMatch = activeSeed ? activeSeed.title.match(/\((\d{4})\)/) : null;
  const cleanSeedYear = seedYearMatch ? seedYearMatch[1] : null;
  const cleanSeedTitle = activeSeed
    ? activeSeed.title.replace(/\s*\(\d{4}\)/, '').trim()
    : '';

  const cleanSeedGenres = activeSeed
    ? activeSeed.genres
        .replace(/\|/g, ', ')
        .split(',')
        .map((g) => g.trim())
        .filter(Boolean)
        .slice(0, 3)
        .join(' • ')
    : '';

  const hasValidSeedOverview =
    Boolean(activeSeed?.overview) &&
    typeof activeSeed?.overview === 'string' &&
    activeSeed.overview.trim().length > 0 &&
    !['nan', 'none', 'null', 'undefined'].includes(activeSeed.overview.trim().toLowerCase());

  return (
    <main className="min-h-screen bg-[#0c0a09] text-stone-100 selection:bg-amber-500/30 selection:text-amber-200">
      {/* Top Cinema Navbar */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-[#0c0a09]/80 border-b border-stone-800/80 px-6 sm:px-10 py-3 transition-all">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 cursor-pointer" onClick={resetToProfile}>
            <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-amber-600 to-amber-400 flex items-center justify-center shadow-lg shadow-amber-500/20">
              <Film size={16} className="text-stone-950 stroke-[2.5]" />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
                CINE<span className="text-amber-400">MATCH</span>
              </span>
              <span className="text-[10px] text-stone-400 font-mono -mt-1 tracking-wider uppercase">
                Hybrid Intelligence
              </span>
            </div>
          </div>

          {/* Centered Search Bar */}
          <div ref={searchRef} className="relative flex-1 max-w-md">
            <div className="relative flex items-center">
              {isSearching ? (
                <Loader2 size={15} className="absolute left-3.5 text-stone-400 animate-spin" />
              ) : (
                <Search size={15} className="absolute left-3.5 text-stone-400" />
              )}
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by title, franchise, or genre..."
                className="w-full bg-stone-900/90 hover:bg-stone-900 focus:bg-stone-950 text-xs text-stone-100 pl-10 pr-4 py-2.5 rounded-full border border-stone-800 focus:border-amber-500/70 outline-none transition-all placeholder:text-stone-500 shadow-inner"
              />
            </div>

            {/* Live Search Dropdown */}
            {searchQuery.trim().length >= 2 && !isSearching && (
              <div className="absolute top-full left-0 right-0 mt-2 bg-stone-900 border border-stone-800 rounded-2xl shadow-2xl p-1.5 z-50 divide-y divide-stone-800/60 max-h-96 overflow-y-auto">
                {searchResults.length > 0 ? (
                  searchResults.map((film) => (
                    <div
                      key={film.movie_id}
                      onClick={() => exploreSimilar(film)}
                      className="flex items-center gap-3 p-2 hover:bg-stone-800/80 rounded-xl transition-colors cursor-pointer group"
                    >
                      <img
                        src={
                          film.poster_url ||
                          `https://placehold.co/100x150/1c1917/a8a29e?text=Film`
                        }
                        alt={film.title}
                        className="w-9 h-13 object-cover rounded-md bg-stone-950 flex-shrink-0"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-stone-200 truncate group-hover:text-amber-400 transition-colors">
                          {film.title}
                        </p>
                        <p className="text-[10px] text-stone-400 truncate">
                          {film.genres.replace(/\|/g, ' • ')}
                        </p>
                      </div>
                      <span className="text-[10px] font-medium text-stone-500 group-hover:text-stone-300 mr-2 flex items-center gap-1">
                        Find Similar →
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="p-4 text-center text-xs text-stone-500">
                    No matching titles found in library
                  </div>
                )}
              </div>
            )}
          </div>

          {/* User Persona Selector */}
          <div className="flex items-center gap-2 bg-stone-900 border border-stone-800 px-3 py-1.5 rounded-full shadow-sm">
            <Users size={14} className="text-amber-400" />
            <select
              value={userId}
              onChange={(e) => {
                setActiveSeed(null);
                setUserId(Number(e.target.value));
              }}
              className="bg-transparent text-xs text-stone-300 outline-none cursor-pointer font-medium"
            >
              <option value={1} className="bg-stone-900 text-stone-200">Taste Profile: Eclectic</option>
              <option value={2} className="bg-stone-900 text-stone-200">Taste Profile: Action & Sci-Fi</option>
              <option value={10} className="bg-stone-900 text-stone-200">Taste Profile: Prestige Drama</option>
              <option value={42} className="bg-stone-900 text-stone-200">Taste Profile: Classic Cinema</option>
              <option value={999} className="bg-stone-900 text-stone-200">Taste Profile: New Explorer (Cold)</option>
            </select>
          </div>
        </div>
      </header>

      {/* Backend Disconnection Banner */}
      {apiError && (
        <div className="max-w-7xl mx-auto px-6 sm:px-10 pt-6">
          <div className="flex items-center gap-3 p-4 bg-amber-950/40 border border-amber-800/80 rounded-2xl text-amber-200 text-xs">
            <AlertCircle size={16} className="shrink-0 text-amber-400" />
            <p className="flex-1">{apiError}</p>
            <button
              onClick={() => (activeSeed ? exploreSimilar(activeSeed) : fetchRecommendations())}
              className="px-3.5 py-1.5 bg-amber-500 hover:bg-amber-400 rounded-full font-semibold text-stone-950 transition-colors"
            >
              Reconnect
            </button>
          </div>
        </div>
      )}

      {/* View Header & Mode Controls */}
      <section className="max-w-7xl mx-auto px-6 sm:px-10 pt-10 pb-6 flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-widest text-amber-400/90 font-semibold">
            <Sparkles size={12} />
            {activeSeed ? 'Contextual Discovery (MMR Filter)' : 'Curated For You'}
          </div>

          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white mt-1">
            {activeSeed ? (
              <span className="flex items-center gap-3 flex-wrap">
                More Like <span className="text-amber-400 italic font-serif">"{cleanSeedTitle}"</span>
                <button
                  onClick={resetToProfile}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-stone-400 hover:text-white bg-stone-800/80 hover:bg-stone-700 px-3 py-1 rounded-full transition-colors border border-stone-700"
                >
                  <X size={13} /> Back to My Recommendations
                </button>
              </span>
            ) : (
              'Top Recommendations'
            )}
          </h2>
        </div>

        {!activeSeed && (
          <div className="inline-flex p-1 bg-stone-900 rounded-full border border-stone-800 shadow-inner self-start md:self-auto">
            {[
              { id: 'hybrid', label: 'Recommended Mix' },
              { id: 'collaborative', label: 'Community Favorites' },
              { id: 'content', label: 'Theme Match' },
              { id: 'compare', label: 'Algorithm Lab' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setMode(tab.id as any)}
                className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${
                  mode === tab.id
                    ? 'bg-amber-500 text-stone-950 font-semibold shadow-md'
                    : 'text-stone-400 hover:text-stone-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        )}
      </section>

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto px-6 sm:px-10 pb-24">
        {loading && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-5">
            {Array.from({ length: 10 }).map((_, i) => (
              <div
                key={i}
                className="flex flex-col bg-stone-900 rounded-xl overflow-hidden border border-stone-800 animate-pulse"
              >
                <div className="aspect-[2/3] bg-stone-800/80" />
                <div className="p-3.5 space-y-2">
                  <div className="h-3.5 bg-stone-800 rounded w-3/4" />
                  <div className="h-2.5 bg-stone-800/60 rounded w-1/2" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Split View with Spotlight + MMR Diversity Steering */}
        {!loading && activeSeed && (
          <div className="flex flex-col lg:flex-row gap-8 items-start animate-fade-in">
            {/* Left Column Spotlight */}
            <aside className="w-full lg:w-80 shrink-0 rounded-2xl bg-gradient-to-b from-stone-900 via-stone-900 to-stone-950 border border-stone-800 p-5 shadow-2xl sticky top-24">
              <div className="flex items-center justify-between gap-2 mb-3.5">
                <span className="text-[10px] font-bold tracking-wider uppercase px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30 flex items-center gap-1">
                  <Film size={10} /> Selected Film
                </span>
                <button
                  onClick={resetToProfile}
                  className="text-xs text-stone-400 hover:text-white transition-colors flex items-center gap-1"
                >
                  <X size={12} /> Clear
                </button>
              </div>

              <div className="aspect-[2/3] w-full rounded-xl overflow-hidden bg-stone-950 border border-stone-800 shadow-xl relative group">
                <img
                  src={
                    activeSeed.poster_url ||
                    `https://placehold.co/500x750/09090b/a1a1aa?text=${encodeURIComponent(cleanSeedTitle)}`
                  }
                  alt={activeSeed.title}
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                />
                {typeof activeSeed.score === 'number' && (
                  <div className="absolute top-2.5 right-2.5 flex items-center gap-1 px-2.5 py-1 rounded-md bg-black/75 backdrop-blur-md border border-white/10 text-white text-xs font-semibold shadow-md">
                    <Star size={12} className="text-amber-400 fill-amber-400" />
                    <span>{activeSeed.score.toFixed(1)}</span>
                  </div>
                )}
              </div>

              <div className="mt-4">
                <div className="flex items-baseline justify-between gap-2">
                  <h3 className="text-lg font-bold text-white leading-snug">
                    {cleanSeedTitle}
                  </h3>
                  {cleanSeedYear && (
                    <span className="text-xs font-mono text-stone-500 shrink-0">
                      {cleanSeedYear}
                    </span>
                  )}
                </div>
                <p className="text-xs text-stone-400 mt-1">
                  {cleanSeedGenres}
                </p>
              </div>

              {/* Interactive MMR Diversity Slider */}
              <div className="mt-5 p-3.5 bg-stone-950/70 border border-stone-800 rounded-xl space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-stone-300 flex items-center gap-1.5">
                    <Sliders size={13} className="text-amber-400" /> MMR Diversity:
                  </span>
                  <span className="font-mono text-amber-400 font-bold">
                    {diversityLambda > 0.8
                      ? 'Direct Sequels'
                      : diversityLambda < 0.5
                      ? 'High Serendipity'
                      : 'Balanced Mix'}
                  </span>
                </div>

                <input
                  type="range"
                  min="0.2"
                  max="1.0"
                  step="0.05"
                  value={diversityLambda}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setDiversityLambda(val);
                    exploreSimilar(activeSeed, val);
                  }}
                  className="w-full h-1.5 bg-stone-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
                <div className="flex justify-between text-[10px] text-stone-500 font-mono">
                  <span>Eclectic (λ=0.2)</span>
                  <span>Pure Clone (λ=1.0)</span>
                </div>
              </div>

              {hasValidSeedOverview && (
                <div className="mt-4 pt-3.5 border-t border-stone-800">
                  <p className="text-[11px] uppercase tracking-wider font-semibold text-stone-400 mb-1.5">
                    Synopsis
                  </p>
                  <p className="text-xs text-stone-300 leading-relaxed bg-stone-950/60 p-3 rounded-xl border border-stone-800/80 max-h-40 overflow-y-auto">
                    {activeSeed.overview}
                  </p>
                </div>
              )}
            </aside>

            {/* Right Column: Re-ranked Grid */}
            <div className="flex-1 min-w-0">
              <div className="mb-4 flex items-center justify-between">
                <p className="text-xs text-stone-400">
                  Showing top {movies.length} nearest thematic & plot matches
                </p>
                <span className="text-[11px] font-mono text-stone-500">
                  λ = {diversityLambda.toFixed(2)}
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-5">
                {movies.map((movie) => (
                  <MovieCard
                    key={movie.movie_id}
                    movie={movie}
                    onSelect={(selected) => exploreSimilar(selected as SearchResult)}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Standard Movie Recommendations Feed */}
        {!loading && !activeSeed && !comparisonData && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-5 animate-fade-in">
            {movies.map((movie) => (
              <MovieCard
                key={movie.movie_id}
                movie={movie}
                onSelect={(selected) => exploreSimilar(selected as SearchResult)}
              />
            ))}
          </div>
        )}

        {/* Algorithm Lab / Side-by-Side Comparison */}
        {!loading && !activeSeed && comparisonData && (
          <div className="flex flex-col gap-10 animate-fade-in">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              {[
                {
                  key: 'content',
                  title: 'Content Similarity',
                  sub: 'FAISS Dense Index',
                  desc: 'Focuses entirely on narrative tropes, genres, and character tags.',
                  border: 'border-sky-800/60',
                  badge: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
                },
                {
                  key: 'collaborative',
                  title: 'Collaborative Filter',
                  sub: 'Matrix Factorization (SVD)',
                  desc: 'Recommends what members with similar viewing habits enjoyed.',
                  border: 'border-purple-800/60',
                  badge: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
                },
                {
                  key: 'hybrid',
                  title: 'Smart Hybrid Ranker',
                  sub: 'Ensemble GBDT',
                  desc: 'Blends collaborative preferences with content affinity for balanced variety.',
                  border: 'border-emerald-800/60',
                  badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
                },
              ].map((m) => (
                <div
                  key={m.key}
                  className={`p-5 rounded-2xl bg-stone-900 border ${m.border} shadow-lg flex flex-col justify-between gap-3`}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-base font-bold text-white">{m.title}</h3>
                      <p className="text-[11px] font-mono text-stone-400 mt-0.5">{m.sub}</p>
                    </div>
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${m.badge}`}>
                      Active
                    </span>
                  </div>
                  <p className="text-xs text-stone-400 border-t border-stone-800 pt-3">
                    {m.desc}
                  </p>
                </div>
              ))}
            </div>

            <div className="flex flex-col gap-8">
              {[0, 1, 2, 3, 4].map((idx) => {
                const rankNum = idx + 1;
                const contentFilm = comparisonData['content']?.[idx] || comparisonData['content_based']?.[idx];
                const collabFilm = comparisonData['collaborative']?.[idx];
                const hybridFilm = comparisonData['hybrid']?.[idx];

                const currentIds = [
                  contentFilm?.movie_id,
                  collabFilm?.movie_id,
                  hybridFilm?.movie_id,
                ].filter(Boolean);
                const isConsensus = currentIds.some((id, i) => currentIds.indexOf(id) !== i);

                return (
                  <div key={idx} className="flex flex-col gap-3">
                    <div className="flex items-center gap-3">
                      <span className="h-6 w-6 rounded-full bg-amber-500 text-stone-950 font-mono text-xs flex items-center justify-center font-bold">
                        {rankNum}
                      </span>
                      <span className="text-xs uppercase tracking-wider font-semibold text-stone-400">
                        Rank #{rankNum} Picks Comparison
                      </span>
                      {isConsensus && (
                        <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          Cross-Model Consensus
                        </span>
                      )}
                      <div className="flex-1 border-t border-stone-800" />
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                      {contentFilm && (
                        <MovieCard
                          movie={contentFilm}
                          rank={rankNum}
                          onSelect={(s) => exploreSimilar(s as SearchResult)}
                        />
                      )}
                      {collabFilm && (
                        <MovieCard
                          movie={collabFilm}
                          rank={rankNum}
                          onSelect={(s) => exploreSimilar(s as SearchResult)}
                        />
                      )}
                      {hybridFilm && (
                        <MovieCard
                          movie={hybridFilm}
                          rank={rankNum}
                          isShared={
                            hybridFilm.movie_id === contentFilm?.movie_id ||
                            hybridFilm.movie_id === collabFilm?.movie_id
                          }
                          onSelect={(s) => exploreSimilar(s as SearchResult)}
                        />
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}