'use client';

import React from 'react';
import { Star, Sparkles } from 'lucide-react';

export interface Movie {
  movie_id: number;
  title: string;
  genres: string;
  score?: number;
  reason?: string;
  poster_url: string;
  overview?: string;
  is_modern?: number;
}

interface MovieCardProps {
  movie: Movie;
  rank?: number;
  isShared?: boolean;
  onSelect?: (movie: Movie) => void;
}

export const MovieCard: React.FC<MovieCardProps> = ({
  movie,
  rank,
  isShared,
  onSelect,
}) => {
  // Normalize pipe/comma delimited genres cleanly
  const genreList = (movie.genres || 'Film')
    .replace(/\|/g, ', ')
    .split(',')
    .map((g) => g.trim())
    .filter(Boolean)
    .slice(0, 2)
    .join(' • ');

  // Extract release year from title if present: "Movie Title (2021)" -> "2021"
  const yearMatch = movie.title.match(/\((\d{4})\)/);
  const releaseYear = yearMatch ? yearMatch[1] : null;
  const cleanTitle = movie.title.replace(/\s*\(\d{4}\)/, '').trim();

  // Extract percentage if reason contains it (e.g., "78% match")
  const matchPercentMatch = movie.reason?.match(/(\d+)%/);
  const matchPercent = matchPercentMatch ? matchPercentMatch[1] : null;

  // Defensive validation to prevent literal 'nan', 'none', or 'null' from rendering
  const hasValidOverview =
    Boolean(movie.overview) &&
    typeof movie.overview === 'string' &&
    movie.overview.trim().length > 0 &&
    !['nan', 'none', 'null', 'undefined'].includes(movie.overview.trim().toLowerCase());

  return (
    <div
      onClick={() => onSelect && onSelect(movie)}
      className="group relative flex flex-col rounded-xl overflow-hidden bg-stone-900 border border-stone-800/80 hover:border-stone-600 transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl hover:shadow-black/50 cursor-pointer"
    >
      {/* Poster Image Container */}
      <div className="relative aspect-[2/3] w-full overflow-hidden bg-stone-950">
        <img
          src={
            movie.poster_url ||
            `https://placehold.co/500x750/09090b/a1a1aa?text=${encodeURIComponent(cleanTitle)}`
          }
          alt={movie.title}
          loading="lazy"
          className="w-full h-full object-cover transition-transform duration-500 ease-out group-hover:scale-105"
        />

        {/* Subtle Dark Gradient Overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-stone-950 via-transparent to-black/30 opacity-70 group-hover:opacity-40 transition-opacity" />

        {/* Top Badges: Rank or Shared indicator */}
        <div className="absolute top-2.5 left-2.5 flex items-center gap-1.5 z-10">
          {rank && (
            <span className="h-6 w-6 rounded-md bg-black/75 backdrop-blur-md border border-white/10 text-white font-mono text-[11px] font-bold flex items-center justify-center shadow-md">
              #{rank}
            </span>
          )}
          {isShared && (
            <span className="px-2 py-0.5 rounded-md bg-amber-500/90 text-stone-950 font-bold text-[10px] tracking-wide uppercase shadow-md flex items-center gap-1 backdrop-blur-sm">
              <Sparkles size={10} /> Top Consensus
            </span>
          )}
        </div>

        {/* Film Rating Pill */}
        {typeof movie.score === 'number' && (
          <div className="absolute top-2.5 right-2.5 flex items-center gap-1 px-2 py-0.5 rounded-md bg-black/70 backdrop-blur-md border border-white/10 text-white text-xs font-semibold shadow-md">
            <Star size={11} className="text-amber-400 fill-amber-400" />
            <span>{movie.score.toFixed(1)}</span>
          </div>
        )}

        {/* Match Percentage Overlay badge (Bottom of poster) */}
        {matchPercent && (
          <div className="absolute bottom-2.5 left-2.5 z-10">
            <span className="px-2 py-0.5 rounded-full bg-emerald-500/90 text-stone-950 text-[10px] font-bold tracking-tight shadow-lg backdrop-blur-sm">
              {matchPercent}% Match
            </span>
          </div>
        )}
      </div>

      {/* Metadata Bottom Pane */}
      <div className="p-3.5 flex flex-col flex-1 justify-between bg-stone-900/95">
        <div>
          <div className="flex items-baseline justify-between gap-1.5">
            <h3
              title={movie.title}
              className="text-sm font-semibold text-stone-100 group-hover:text-amber-400 transition-colors line-clamp-1 leading-snug"
            >
              {cleanTitle}
            </h3>
            {releaseYear && (
              <span className="text-[11px] font-mono text-stone-500 shrink-0">
                {releaseYear}
              </span>
            )}
          </div>
          <p className="text-[11px] text-stone-400 mt-1 truncate">
            {genreList}
          </p>

          {/* Synopsis / Movie Description (Protected against 'nan') */}
          {hasValidOverview && (
            <p className="text-[11px] text-stone-400/90 mt-2 line-clamp-2 leading-relaxed font-normal">
              {movie.overview}
            </p>
          )}
        </div>

        {/* Recommendation Reason Pill */}
        {movie.reason && !matchPercent && (
          <div className="mt-2.5 pt-2 border-t border-stone-800 text-[10px] text-stone-400 line-clamp-1">
            {movie.reason}
          </div>
        )}
      </div>
    </div>
  );
};